"""MolCore processing for the widget's localisation and display controls."""

import numpy as np
from molcore import (
    Acquisition,
    Dataset,
    ExecutionContext,
    FilterSettings,
    MissingSource,
    PartitionSelection,
    RenderSettings,
)
from molcore.filtering import filter_results
from molcore.rendering import render_localizations


def excluded_localisation_ids(
    snapshot, parts, channel, metric, lower, upper, *,
    cancel_event=None, progress_callback=None,
):
    """Find excluded IDs across all acquisitions of a logical channel."""
    context = ExecutionContext(cancel_event=cancel_event)
    lower = np.nextafter(float(lower), np.inf)
    upper = np.nextafter(float(upper), -np.inf)
    excluded = []
    for index, (view, offset) in enumerate(parts, start=1):
        context.check_cancelled()
        acquisition_id = view.acquisition.id
        dataset = snapshot.localizations.query(
            acquisition_id=acquisition_id, channel=channel, space="raw"
        )
        ids = dataset["id"]
        if ids.size:
            bounds = lower, upper
            if metric == "frame":
                # The spin boxes show concatenated logical indices. MolCore
                # stores the corresponding camera frame within each recording.
                frames = view.source_frames
                first = max(0, int(np.ceil(lower)) - offset)
                last = min(len(frames) - 1, int(np.floor(upper)) - offset)
                bounds = (frames[first], frames[last]) if first <= last else (1, 0)
            if bounds[0] <= bounds[1]:
                result = filter_results(
                    snapshot,
                    PartitionSelection(acquisition_id, channel),
                    settings=FilterSettings(
                        metric=metric, minimum=bounds[0], maximum=bounds[1]
                    ),
                    space="raw",
                    context=context,
                )
                ids = ids[~np.isin(ids, result.selected_ids)]
            excluded.append(ids)
        if progress_callback is not None:
            progress_callback.emit(100 * index // len(parts))
    context.check_cancelled()
    return np.concatenate(excluded) if excluded else np.empty(0, dtype=np.uint64)


def render_localisations(
    locs, image_shape, blur_method=None, min_blur_width=1,
    pixel_size=1, progress_callback=None, oversampling=20, *, cancel_event=None,
):
    """Render a detached pool of localisation rows in displayed channel pixels."""
    height, width = image_shape[-2:]
    context = ExecutionContext(
        cancel_event=cancel_event,
        progress=(
            None if progress_callback is None
            else lambda update: progress_callback.emit(int(update.percent))
        ),
    )
    context.check_cancelled()
    with Dataset() as model:
        # Picasso computes one median precision for convolution. A pooled
        # partition preserves that behavior when a channel spans acquisitions.
        acquisition = Acquisition(
            MissingSource((1, height, width), np.uint16, "Rendering uses coordinates only")
        )
        model.add_acquisition(acquisition)
        (channel,) = acquisition.channels
        if len(locs):
            columns = {
                field: locs[field]
                for field in ("x", "y", "lpx", "lpy")
                if field in locs.dtype.names
            }
            model.localizations.add(
                acquisition_id=acquisition.id,
                channel=channel,
                frame=np.zeros(len(locs), dtype=np.int64),
                **columns,
            )
        result = render_localizations(
            model.snapshot(),
            PartitionSelection(acquisition.id, channel),
            settings=RenderSettings(
                oversampling=oversampling,
                blur_method=blur_method or "none",
                min_blur_width=min_blur_width,
            ),
            space="raw",
            viewport=((0, 0), (height, width)),
            context=context,
        )
    return result.image, pixel_size, oversampling
