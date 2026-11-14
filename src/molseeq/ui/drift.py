"""Estimate MolCore drift and stage matching pixel and localisation corrections."""

from concurrent.futures import CancelledError
from dataclasses import asdict
from functools import partial

import numpy as np
from molcore import (
    DriftSettings,
    ExecutionContext,
    StaleResultError,
    undrifting,
)
from molcore.staging import state_token
from scipy.ndimage import shift

from molseeq.session import ImageCorrection
from molseeq.ui.worker import Worker


def compute_drift(session, names, channel, settings, cancel_event, progress_callback=None):
    """Compute complete corrections from a detached session with borrowed sources."""
    corrections = {}
    last_progress = -1
    for index, name in enumerate(names):
        def progress(percent):
            nonlocal last_progress
            value = int((index * 100 + percent) / len(names))
            if progress_callback is not None and value > last_progress:
                progress_callback.emit(value)
                last_progress = value

        context = ExecutionContext(
            cancel_event=cancel_event,
            progress=lambda update: progress(update.percent / 2),
        )
        context.check_cancelled()
        model = session.dataset(name)
        selections = {
            view.acquisition.id: channel
            for view, _ in session.channel_parts(name, channel)
        }
        try:
            estimates = undrifting.undrift_many(
                model.snapshot(), selections, settings=settings, context=context
            )
        except ValueError as error:
            raise ValueError(f"Could not estimate drift for {name}: {error}") from error

        images = {}
        moves = []
        sources = {
            target: session.working_source(name, target)
            for target in session.images[name]
        }
        total_frames = sum(len(source) for source in sources.values())
        completed_frames = 0
        for target, source in sources.items():
            pixels = np.empty(source.shape, dtype=source.dtype)
            for view, offset in session.channel_parts(name, target):
                acquisition = view.acquisition.id
                estimate = estimates[acquisition]
                dx, dy = (
                    np.interp(view.source_frames, estimate.frames, values)
                    for values in estimate.displacement.T
                )
                for frame, (x, y) in enumerate(zip(dx, dy, strict=True), start=offset):
                    context.check_cancelled()
                    if x or y:
                        shift(
                            source[frame], (-y, -x), output=pixels[frame],
                            order=3, mode="constant", cval=0,
                        )
                    else:
                        pixels[frame] = source[frame]
                    completed_frames += 1
                    progress(50 + 50 * completed_frames / total_frames)
                rows = model.localizations.query(
                    acquisition_id=acquisition, channel=target, space="raw"
                )
                indices = view.channel_indices(rows["frame"])
                changed = (dx[indices] != 0) | (dy[indices] != 0)
                indices = indices[changed]
                if len(indices):
                    moves.append((
                        rows["id"][changed],
                        rows["x"][changed] - dx[indices],
                        rows["y"][changed] - dy[indices],
                    ))
            images[target] = pixels
        operation = {
            "method": "drift_correction",
            "settings": asdict(settings),
            "reference_channel": channel,
            "interpolation": "scipy_cubic",
            "border_value": 0,
            "camera_frame_edge_policy": "nearest_estimate",
            "estimates": {
                acquisition: {
                    "camera_frame_range": [int(estimate.frames[0]), int(estimate.frames[-1])],
                    "backend_version": estimate.backend_version,
                }
                for acquisition, estimate in estimates.items()
            },
        }
        corrections[name] = ImageCorrection(images, moves, operation)
    return corrections


class DriftCorrectionMixin:
    def _start_drift(self, settings, progress_bar):
        selected = self.gui.undrift_dataset_selector.currentText()
        channel = self.gui.undrift_channel_selector.currentText().lower()
        datasets = list(self.session.images) if selected == "All Datasets" else [selected]
        names = []
        for name in datasets:
            if not name:
                continue
            _, count, fitted = self.get_loc_dict(name, channel)
            if count and fitted:
                names.append(name)
            else:
                self.molseeq_notification(f"No fitted localisations found for {name} ({channel})")
        if not names:
            return

        revision = self.session.revision
        expected = {name: state_token(self.session.dataset(name)) for name in names}
        snapshot = self.session.image_snapshot()
        self.update_ui(init=True)
        self.worker = Worker(compute_drift, snapshot, names, channel, settings, self.stop_event)
        self.worker.signals.progress.connect(
            partial(self.molseeq_progress, progress_bar=progress_bar)
        )
        self.worker.signals.result.connect(
            partial(self._publish_drift, revision=revision, expected=expected)
        )
        self.worker.signals.error.connect(self._report_drift_error)
        self.worker.signals.finished.connect(self.update_ui)
        self.threadpool.start(self.worker)

    def _publish_drift(self, corrections, *, revision, expected):
        try:
            self.session.publish_image_correction(
                corrections, revision=revision, expected=expected, cancel_event=self.stop_event
            )
        except CancelledError:
            return
        except (ValueError, StaleResultError) as error:
            self.molseeq_notification(str(error))
            return
        if self.active_dataset in corrections:
            self.update_active_image(dataset=self.active_dataset, channel=self.active_channel)
        self.draw_localisations(update_vis=True)

    def _report_drift_error(self, error):
        if not isinstance(error[1], CancelledError):
            self.molseeq_notification(str(error[1]))

    def aim_undrift(self):
        try:
            settings = DriftSettings(
                method="aim",
                segment_frames=self.gui.aim_segmentation.value(),
                intersect_distance=self.gui.aim_intersect_d.value(),
                search_radius=self.gui.aim_roi_r.value(),
            )
        except ValueError as error:
            self.molseeq_notification(str(error))
            return
        self._start_drift(settings, self.gui.aim_progressbar)

    def rcc_undrift(self):
        self._start_drift(
            DriftSettings(method="rcc", segment_frames=self.gui.undrift_segmentation.value()),
            self.gui.undrift_progressbar,
        )
