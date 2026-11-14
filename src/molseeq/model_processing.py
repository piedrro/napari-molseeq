"""Detached MolCore processing requests for the widget's CPU workflows."""

from dataclasses import dataclass, replace

import numpy as np
from molcore import (
    CameraSettings,
    ExecutionContext,
    LocalisationSettings,
    PartitionSelection,
    localisation,
)
from molcore.snapshots import DatasetSnapshot
from molcore.types.acquisition import ChannelView
from molcore.types.images import Source
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LocalisationJob:
    dataset: DatasetSnapshot
    view: ChannelView
    offset: int
    frame: int | None = None
    source: Source | None = None
    seeds: np.recarray | None = None


def _fit_gpufit(job, columns, box_size, context):
    from picasso.fitting.precision import localization_precision

    from pygpufit import gpufit

    view = job.view
    source = job.source if job.source is not None else view
    frames = view.channel_indices(columns["frame"])
    x = np.rint(columns["x"]).astype(int)
    y = np.rint(columns["y"]).astype(int)
    half = box_size // 2
    spots = np.empty((len(frames), box_size, box_size), dtype=np.float32)
    for frame in np.unique(frames):
        context.check_cancelled()
        image = source[frame]
        for i in np.flatnonzero(frames == frame):
            spots[i] = image[y[i] - half:y[i] + half + 1, x[i] - half:x[i] + half + 1]
    pixels = spots.reshape(len(frames), -1) - np.float32(100)
    peak = pixels.argmax(axis=1)
    background = pixels.min(axis=1)
    initial = np.ones((len(frames), 6), dtype=np.float32)
    initial[:, 0] = np.maximum(1, pixels.max(axis=1) - background)
    initial[:, 1] = peak % box_size
    initial[:, 2] = peak // box_size
    initial[:, 5] = background
    params, states, _, _, _ = gpufit.fit(
        pixels, None, gpufit.ModelID.GAUSS_2D_ELLIPTIC, initial,
        tolerance=1e-2, max_number_iterations=20,
    )
    context.check_cancelled()
    keep = (states == 0) & np.all(np.isfinite(params), axis=1)
    keep &= (params[:, 0] > 0) & np.all(params[:, 3:5] > 0, axis=1)
    if not keep.any():
        raise ValueError("No Gpufit localisation fits converged")
    amplitude, dx, dy, sx, sy, bg = params[keep].T
    photons = amplitude * (2 * np.pi * sx * sy)
    return {
        "frame": columns["frame"][keep],
        "x": x[keep] + dx - half,
        "y": y[keep] + dy - half,
        "photons": photons,
        "sx": sx,
        "sy": sy,
        "bg": bg,
        "lpx": localization_precision(photons, sx, sy, bg, em=False),
        "lpy": localization_precision(photons, sy, sx, bg, em=False),
        "ellipticity": 1 - np.minimum(sx, sy) / np.maximum(sx, sy),
        "net_gradient": columns["net_gradient"][keep],
    }


def _select_candidates(columns, settings, roi, remove_overlapping):
    keep = np.ones(len(columns["frame"]), dtype=bool)
    if roi is not None:
        (y0, x0), (y1, x1) = roi
        x, y = columns["x"], columns["y"]
        keep &= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    if remove_overlapping:
        for frame in np.unique(columns["frame"][keep]):
            indices = np.flatnonzero(keep & (columns["frame"] == frame))
            xy = np.column_stack(
                (columns["x"][indices], columns["y"][indices])
            )
            radius = np.nextafter(float(settings.box_size), 0)
            pairs = cKDTree(xy).query_pairs(radius, output_type="ndarray")
            keep[indices[pairs.ravel()]] = False
    return {name: values[keep] for name, values in columns.items()}


def localise_channel(
    job: LocalisationJob,
    settings: LocalisationSettings,
    *,
    detect: bool,
    fit: bool,
    gpu_fit: bool = False,
    roi=None,
    remove_overlapping=False,
    cancel_event=None,
):
    """Compute detached results in the legacy channel-local frame clock."""
    view = job.view
    frame_range = None
    if job.frame is not None:
        camera_frame = int(view.source_frames[job.frame - job.offset])
        frame_range = (camera_frame, camera_frame + 1)
    snapshot = (
        job.dataset.snapshot()
        if hasattr(job.dataset, "snapshot")
        else job.dataset
    )
    context = ExecutionContext(cancel_event=cancel_event)
    options = {"context": context}
    if job.source is not None:
        options["source"] = job.source
    found = localisation.detect(
        snapshot,
        PartitionSelection(view.acquisition.id, view.name),
        settings=settings,
        frame_range=frame_range if detect else (0, 0),
        **options,
    )
    if detect:
        columns = _select_candidates(
            found.columns, settings, roi, remove_overlapping
        )
    else:
        seeds = job.seeds
        columns = {
            "x": seeds.x.copy(),
            "y": seeds.y.copy(),
            "frame": view.camera_frames(seeds.frame - job.offset),
            "net_gradient": (
                seeds.net_gradient.copy()
                if "net_gradient" in seeds.dtype.names
                else np.zeros(len(seeds))
            ),
        }
    if fit and len(columns["frame"]):
        half = settings.box_size // 2
        x, y = columns["x"], columns["y"]
        if gpu_fit:
            x, y = np.rint(x), np.rint(y)
        valid = (
            (x >= half)
            & (x < view.shape[2] - half)
            & (y >= half)
            & (y < view.shape[1] - half)
        )
        if not valid.any():
            raise ValueError("No localisation has a complete fitting box")
        columns = {name: values[valid] for name, values in columns.items()}
        if gpu_fit:
            columns = _fit_gpufit(job, columns, settings.box_size, context)
        else:
            found = replace(found, columns=columns, frame_range=frame_range)
            fitted = localisation.fit(
                snapshot, found, camera=CameraSettings(100, 1, 1), **options,
            )
            if fitted.failed_fits and not len(fitted.columns["frame"]):
                raise ValueError("No localisation fits converged")
            columns = dict(fitted.columns)
    columns["frame"] = view.channel_indices(columns["frame"]) + job.offset
    return np.rec.fromarrays(tuple(columns.values()), names=tuple(columns))


def localise_channels(jobs, settings, *, progress_callback, **options):
    """Run prepared partition jobs on the worker and group their detached rows."""
    parts = {}
    for index, (name, job) in enumerate(jobs, 1):
        rows = localise_channel(job, settings, **options)
        parts.setdefault(name, []).append(rows)
        progress_callback.emit(index * 100 // len(jobs))
    result = {}
    for name, chunks in parts.items():
        nonempty = [rows for rows in chunks if len(rows)]
        result[name] = np.concatenate(nonempty or chunks[:1]).view(np.recarray)
    return result
