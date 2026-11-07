"""Detached MolCore processing requests for the widget's CPU workflows."""

from dataclasses import dataclass, replace
from inspect import signature

import numpy as np
from molcore import CameraSettings, Dataset, LocalisationSettings
from molcore import localisation
from molcore.types.acquisition import ChannelView
from molcore.types.images import Source
from scipy.spatial import cKDTree

SUPPORTS_WORKING_LOCALISATION = (
    "source" in signature(localisation.detect).parameters
)


@dataclass(frozen=True)
class LocalisationJob:
    dataset: Dataset
    view: ChannelView
    offset: int
    frame: int | None = None
    source: Source | None = None
    seeds: np.recarray | None = None


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
    options = {"cancel_event": cancel_event}
    if job.source is not None:
        options["source"] = job.source
    found = localisation.detect(
        job.dataset,
        view.acquisition.id,
        view.name,
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
        valid = (
            (x >= half)
            & (x < view.shape[2] - half)
            & (y >= half)
            & (y < view.shape[1] - half)
        )
        if not valid.any():
            raise ValueError("No localisation has a complete fitting box")
        found = replace(
            found,
            columns={name: values[valid] for name, values in columns.items()},
            frame_range=frame_range,
        )
        fitted = localisation.fit(
            job.dataset,
            found,
            camera=CameraSettings(100, 1, 1),
            apply=False,
            **options,
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
