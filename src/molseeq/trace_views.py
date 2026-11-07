"""Bind trace results to MolCore camera frames and project legacy metric names."""

from uuid import uuid4

import numpy as np
from molcore import ApertureShape, ArraySource, PhotonSequence, PhotonUnit
from molcore import photometry
from molcore.staging import clone_experiment

_METHOD = "molseeq_aperture"


def discard_trace_runs(experiment, dataset_ids=None):
    changed = False
    for identifier, model in experiment.datasets.items():
        if dataset_ids is not None and identifier not in dataset_ids:
            continue
        for sequence in model.photon_sequences.query(include_stale=True):
            if sequence.method == _METHOD:
                model.photon_sequences.remove(sequence.id)
                changed = True
    return changed


def compute_fixed_traces(
    session, locs, options, *, box_size=None, progress=None, cancel_event=None
):
    """Measure fixed sites with MolCore and retain MolSeeq's existing metric names."""
    size, shape, buffer, width, global_background, fit_psf = options
    settings = photometry.FixedApertureSettings(
        spot_size=size,
        shape=ApertureShape(shape.lower()),
        buffer=buffer,
        background_width=width,
        global_background=global_background,
        psf_box_size=box_size if fit_psf else None,
    )
    sites = {
        str(index): (float(loc.x), float(loc.y))
        for index, loc in enumerate(locs)
    }
    jobs = [
        (
            name,
            channel,
            session.dataset(name),
            view,
            offset,
            session.working_source(name, channel),
        )
        for name, channels in session.images.items()
        for channel in channels
        for view, offset in session.channel_parts(name, channel)
        if len(view)
    ]
    output = {}
    for job, (name, channel, dataset, view, offset, source) in enumerate(jobs):
        working = (
            ArraySource(source.array[offset : offset + len(view)])
            if isinstance(source, ArraySource)
            else None
        )
        result = photometry.stationary_site_traces(
            dataset,
            view.acquisition.id,
            channel,
            sites,
            settings=settings,
            source=working,
            apply=False,
            cancel_event=cancel_event,
            progress=(
                None
                if progress is None
                else lambda update: progress(
                    int((job + update.percent / 100) * 100 / len(jobs))
                )
            ),
        )
        channel_spots = output.setdefault(name, {}).setdefault(channel, {})
        prefixes = {
            "signal": "",
            "local_window": "_local_bg",
            "masked_local_ring": "_masked_local_bg",
        }
        if global_background:
            prefixes.update(
                global_background="_global_bg",
                masked_global_background="_masked_global_bg",
            )
        for sequence in result.sequences:
            index = int(sequence.molecule_id)
            x, y = sites[sequence.molecule_id]
            count = len(sequence.frames)
            columns = {
                "spot_x": np.full(count, x),
                "spot_y": np.full(count, y),
                "spot_cx": np.full(count, y),
                "spot_cy": np.full(count, x),
                "spot_size": np.full(count, size),
            }
            for prefix, suffix in prefixes.items():
                for metric in ("mean", "median", "sum", "max", "std"):
                    samples = sequence.measurements[f"{prefix}_{metric}"]
                    if suffix and metric == "sum":
                        samples = (
                            sequence.measurements[f"{prefix}_mean"] * size**2
                        )
                    columns[f"spot_{metric}{suffix}"] = samples
            if fit_psf:
                for target, metric in {
                    "spot_photons": "amplitude",
                    "spot_photons_local_bg": "background",
                    "spot_photons_masked_local_bg": "background",
                    "spot_sx": "sx",
                    "spot_sy": "sy",
                    "spot_lpx": "lpx",
                    "spot_lpy": "lpy",
                    "spot_likelihood": "likelihood",
                }.items():
                    columns[target] = sequence.measurements[f"psf_{metric}"]
                gradient = (
                    locs[index].net_gradient
                    if "net_gradient" in locs.dtype.names
                    else 0
                )
                columns["spot_net_gradient"] = np.full(count, gradient)
            parts = channel_spots.setdefault(index, {})
            for key, values in columns.items():
                parts.setdefault(key, []).append(values)
    for channels in output.values():
        for spots in channels.values():
            for index, columns in spots.items():
                spots[index] = {
                    key: np.concatenate(parts)
                    for key, parts in columns.items()
                }
    return output


def publish_traces(session, traces):
    """Publish a complete trace run atomically across the selected rounds."""
    staged = clone_experiment(session.experiment)
    discard_trace_runs(staged)
    run_id = str(uuid4())
    for name in session.images:
        model = staged.datasets[session.dataset(name).id]
        for channel, spots in traces.get(name, {}).items():
            count = len(session.channel(name, channel))
            for index, metrics in spots.items():
                values = {
                    key: samples
                    for key, samples in metrics.items()
                    if isinstance(samples, np.ndarray)
                    and samples.dtype.kind in "biuf"
                }
                if any(
                    samples.shape != (count,) for samples in values.values()
                ):
                    raise ValueError(
                        "Trace samples must match the logical channel frame count"
                    )
                export_offset = 0
                for view, offset in session.channel_parts(name, channel):
                    alternating = view.acquisition.mode == "alex"
                    export_frames = (
                        view.source_frames // 2
                        if alternating
                        else view.source_frames
                    )
                    indices = export_frames + export_offset
                    camera_count = len(view.acquisition.source)
                    export_offset += (
                        (camera_count + 1) // 2
                        if alternating
                        else camera_count
                    )
                    if not len(view):
                        continue
                    part = slice(offset, offset + len(view))
                    model.photon_sequences.add(
                        PhotonSequence(
                            acquisition_id=view.acquisition.id,
                            channel=channel,
                            frames=view.source_frames,
                            photons=values["spot_sum"][part],
                            molecule_id=f"molseeq:{index}",
                            unit=PhotonUnit.IMAGE_INTENSITY,
                            method=_METHOD,
                            run_id=run_id,
                            metadata={
                                "spot_index": int(index),
                                "display_offset": offset,
                            },
                            measurements={
                                **{
                                    key: samples[part]
                                    for key, samples in values.items()
                                },
                                "_export_frames": indices,
                            },
                            depends_on_transforms=True,
                        )
                    )
    session.experiment = staged
    session.revision += 1
    session.trace_revision += 1


def trace_view(session):
    """Return detached, ordered trace arrays for the current plot and exporters."""
    result = {}
    for name in session.images:
        model = session.dataset(name)
        channels = {}
        for sequence in model.photon_sequences.query():
            if sequence.method != _METHOD:
                continue
            spot = sequence.metadata["spot_index"]
            channels.setdefault(sequence.channel, {}).setdefault(
                spot, []
            ).append(sequence)
        if not channels:
            continue
        result[name] = {}
        for channel, spots in channels.items():
            result[name][channel] = {}
            for index, segments in sorted(spots.items()):
                segments.sort(
                    key=lambda segment: segment.metadata["display_offset"]
                )
                metrics = {
                    key: np.concatenate(
                        [segment.measurements[key] for segment in segments]
                    )
                    for key in segments[0].measurements
                }
                metrics["_camera_frames"] = np.concatenate(
                    [segment.frames for segment in segments]
                )
                metrics["_acquisition_ids"] = tuple(
                    segment.acquisition_id
                    for segment in segments
                    for _ in segment.frames
                )
                for key in ("gap_label", "sequence_label"):
                    metrics[key] = model.metadata.get(key)
                result[name][channel][index] = metrics
    return result


def align_trace_samples(values, source, reference):
    """Align one channel's measurements to matching camera samples."""
    if "_camera_frames" not in source or "_camera_frames" not in reference:
        return np.asarray(values)
    source_keys = zip(source["_acquisition_ids"], source["_camera_frames"], strict=True)
    reference_keys = zip(
        reference["_acquisition_ids"], reference["_camera_frames"], strict=True
    )
    indexed = dict(zip(source_keys, values, strict=True))
    return np.asarray([indexed.get(key, np.nan) for key in reference_keys])


def expand_export_trace(values, trace):
    """Pad missing cycles inside a trace for rectangular legacy export formats."""
    frames = trace.get("_export_frames")
    if frames is None or not len(frames):
        return np.asarray(values, dtype=float)
    expanded = np.full(int(frames[-1]) + 1, np.nan)
    expanded[frames] = values
    return expanded


def stack_export_traces(traces, *, axis=0):
    """Make a rectangular export with NaN for missing samples."""
    size = max(map(len, traces), default=0)
    if not traces:
        return np.empty((0, 0))
    columns = [
        np.pad(
            np.asarray(trace, dtype=float),
            (0, size - len(trace)),
            constant_values=np.nan,
        )
        for trace in traces
    ]
    return np.stack(columns, axis=axis)
