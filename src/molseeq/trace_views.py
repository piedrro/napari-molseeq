"""Bind trace results to MolCore camera frames and project legacy metric names."""

from uuid import uuid4
from dataclasses import asdict, replace

import numpy as np
from molcore import ApertureShape, ArraySource, PhotonSequence, PhotonUnit
from molcore import photometry
from molcore.staging import clone_experiment, experiment_state_token
from molcore import StaleResultError
from molseeq import __version__

_METHOD = "molseeq_aperture"


def discard_trace_runs(experiment, dataset_ids=None):
    changed = False
    for identifier, model in experiment.datasets.items():
        if dataset_ids is not None and identifier not in dataset_ids:
            continue
        ids = [
            sequence.id
            for sequence in model.photon_sequences.query()
            if sequence.method == _METHOD
        ]
        if ids:
            model.photon_sequences.mark_stale(ids)
            changed = True
    return changed


class TraceResults(dict):
    """Legacy plotting columns accompanied by the original MolCore trace records."""

    def __init__(self, session, settings):
        super().__init__()
        self.records = []
        self.run_id = str(uuid4())
        self.settings = asdict(settings)
        self.session_revision = session.revision
        self.expected = experiment_state_token(session.experiment)


def _legacy_aperture_columns(sequence):
    settings = sequence.metadata.get("extraction", {}).get(
        "settings", sequence.metadata["settings"]
    )
    site = sequence.metadata["legacy_spot"]
    size = settings["spot_size"]
    count = len(sequence.frames)
    columns = {
        "spot_x": np.full(count, site["x"]),
        "spot_y": np.full(count, site["y"]),
        "spot_cx": np.full(count, site["y"]),
        "spot_cy": np.full(count, site["x"]),
        "spot_size": np.full(count, size),
    }
    prefixes = {
        "signal": "",
        "local_window": "_local_bg",
        "masked_local_ring": "_masked_local_bg",
    }
    if settings["global_background"]:
        prefixes.update(
            global_background="_global_bg",
            masked_global_background="_masked_global_bg",
        )
    for prefix, suffix in prefixes.items():
        for metric in ("mean", "median", "sum", "max", "std"):
            samples = sequence.measurements[f"{prefix}_{metric}"]
            if suffix and metric == "sum":
                samples = sequence.measurements[f"{prefix}_mean"] * size**2
            columns[f"spot_{metric}{suffix}"] = samples
    if settings["psf_box_size"] is not None:
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
        columns["spot_net_gradient"] = np.full(
            count,
            np.nan if site["net_gradient"] is None else site["net_gradient"],
        )
    return columns


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
    identifiers = session.bounding_box_ids()
    anchored = bool(identifiers)
    if anchored:
        if len(identifiers) != len(locs):
            raise ValueError(
                "Bounding box selection changed before extraction"
            )
        current = session.bounding_boxes["localisations"]
        if not np.array_equal(current.x, locs.x) or not np.array_equal(
            current.y, locs.y
        ):
            raise ValueError(
                "Bounding box positions changed before extraction"
            )
    else:
        identifiers = tuple(str(index) for index in range(len(locs)))
    sites = {
        identifier: (float(loc.x), float(loc.y))
        for identifier, loc in zip(identifiers, locs, strict=True)
    }
    spot_indices = {
        identifier: index for index, identifier in enumerate(identifiers)
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
    output = TraceResults(session, settings)
    for job, (name, channel, dataset, view, offset, source) in enumerate(jobs):
        working = (
            ArraySource(source.array[offset : offset + len(view)])
            if isinstance(source, ArraySource)
            else view
            if anchored
            else None
        )
        measure = (
            photometry.bounding_box_traces
            if anchored
            else photometry.stationary_site_traces
        )
        result = measure(
            dataset,
            view.acquisition.id,
            channel,
            tuple(sites) if anchored else sites,
            settings=settings,
            source=working,
            apply=False,
            cancel_event=cancel_event,
            **(
                {
                    "source_metadata": session.working_image_provenance(
                        name, channel
                    )
                }
                if anchored
                else {}
            ),
            progress=(
                None
                if progress is None
                else lambda update: progress(
                    int((job + update.percent / 100) * 100 / len(jobs))
                )
            ),
        )
        channel_spots = output.setdefault(name, {}).setdefault(channel, {})
        for sequence in result.sequences:
            index = spot_indices[sequence.molecule_id]
            x, y = sites[sequence.molecule_id]
            if sequence.bounding_box_id is not None:
                x, y = (
                    sequence.metadata["bounding_box"]["x"],
                    sequence.metadata["bounding_box"]["y"],
                )
            gradient = (
                float(locs[index].net_gradient)
                if "net_gradient" in locs.dtype.names
                else 0.0
            )
            if not np.isfinite(gradient):
                gradient = None
            export_offset = sum(
                (len(previous.acquisition.source) + 1) // 2
                if previous.acquisition.mode == "alex"
                else len(previous.acquisition.source)
                for previous, part_offset in session.channel_parts(
                    name, channel
                )
                if part_offset < offset
            )
            export_frames = (
                sequence.frames // 2
                if view.acquisition.mode == "alex"
                else sequence.frames
            )
            output.records.append(
                (
                    dataset.id,
                    replace(
                        sequence,
                        method=_METHOD,
                        run_id=output.run_id,
                        metadata={
                            **sequence.metadata,
                            "spot_index": index,
                            "display_offset": offset,
                            "application": "molseeq",
                            "application_version": __version__,
                            "legacy_spot": {
                                "x": x,
                                "y": y,
                                "net_gradient": gradient,
                            },
                        },
                        measurements={
                            **sequence.measurements,
                            "_export_frames": export_frames + export_offset,
                        },
                    ),
                )
            )
            columns = _legacy_aperture_columns(output.records[-1][1])
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
    if isinstance(traces, TraceResults):
        if (
            session.revision != traces.session_revision
            or experiment_state_token(session.experiment) != traces.expected
        ):
            raise StaleResultError("Trace inputs changed before publication")
        staged = clone_experiment(session.experiment)
        discard_trace_runs(staged)
        for dataset_id, sequence in traces.records:
            staged.datasets[dataset_id].photon_sequences.add(sequence)
        for model in staged.datasets.values():
            records = [
                sequence
                for dataset_id, sequence in traces.records
                if dataset_id == model.id
            ]
            if records:
                model.processing.settings.setdefault("photon_sequences", {})[
                    traces.run_id
                ] = {
                    "version": 2,
                    "application": "molseeq",
                    "settings": traces.settings,
                    "sequence_ids": [sequence.id for sequence in records],
                    "bounding_box_ids": sorted(
                        {
                            sequence.bounding_box_id
                            for sequence in records
                            if sequence.bounding_box_id
                        }
                    ),
                }
        session.experiment = staged
        session.revision += 1
        session.trace_revision += 1
        return
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
    box_indices = {
        identifier: index
        for index, identifier in enumerate(session.bounding_box_ids())
    }
    for name in session.images:
        model = session.dataset(name)
        channels = {}
        for sequence in model.photon_sequences.query():
            if sequence.method != _METHOD:
                continue
            spot = box_indices.get(
                sequence.bounding_box_id, sequence.metadata["spot_index"]
            )
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
                if "legacy_spot" in segments[0].metadata:
                    projections = [
                        _legacy_aperture_columns(segment)
                        for segment in segments
                    ]
                    metrics.update(
                        {
                            key: np.concatenate(
                                [projection[key] for projection in projections]
                            )
                            for key in projections[0]
                        }
                    )
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
    source_keys = zip(
        source["_acquisition_ids"], source["_camera_frames"], strict=True
    )
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
