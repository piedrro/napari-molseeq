import traceback
from functools import partial

import numpy as np
from molcore import (
    CoordinateSpace,
    ExecutionContext,
    PartitionSelection,
    SequenceSelection,
    StaleResultError,
    TrackingSettings,
    tracking,
)
from molcore.staging import clone_dataset, state_token, transact

from molseeq.ui.worker import Worker


def _track_model(model, identifier, is_sequence, channel, settings, remove_unlinked,
                 cancel_event, progress_callback):
    snapshot = model
    model = clone_dataset(snapshot)
    options = {
        "settings": settings,
        "context": ExecutionContext(
            cancel_event=cancel_event,
            progress=lambda update: progress_callback.emit(int(update.percent)),
            num_cpus=1,
        ),
    }
    if is_sequence:
        result = tracking.track_sequence(snapshot, SequenceSelection(identifier, channel), **options)
        receipt = model.apply(result)
        parts = model.sequences[identifier].acquisitions
        rows = model.tracks.query(
            sequence_id=identifier, acquisition_id=[item.id for item in parts],
            channel=channel, space="raw",
        )
    else:
        result = tracking.track(snapshot, PartitionSelection(identifier, channel), **options)
        receipt = model.apply(result)
        parts = [model.acquisitions[identifier]]
        rows = model.tracks.query(acquisition_id=identifier, channel=channel, space="raw")

    frame = np.empty(len(rows["id"]), dtype=np.int64)
    offset = 0
    for acquisition in parts:
        view = acquisition.channels[channel]
        mask = rows["acquisition_id"] == acquisition.id
        frame[mask] = view.channel_indices(rows["frame"][mask]) + offset
        offset += len(view)
    unlinked = np.empty(0, dtype=np.uint64)
    if remove_unlinked:
        ids = model.localizations.query(
            acquisition_id=[item.id for item in parts], channel=channel, space="raw"
        )["id"]
        unlinked = np.setdiff1d(ids, rows["id"])
    labels = np.searchsorted(receipt.track_ids, rows["track_id"])
    return result, np.column_stack((labels + 1, frame, rows["y"], rows["x"])), unlinked


class TrackingMixin:

    def tracking_finished(self):

        self.draw_localisations()
        self.update_ui()

    def _publish_tracks(self, result, dataset, channel, expected, revision):
        if self.stop_event.is_set() or self.session.revision != revision:
            return
        detached, rows, unlinked = result
        model = self.session.dataset(dataset)
        try:
            def publish(staged):
                staged.apply(detached)
                if len(unlinked):
                    staged.localizations.delete(unlinked)
            transact(model, publish, expected=expected)
        except (ValueError, StaleResultError) as error:
            self.molseeq_notification(str(error))
            return
        self.session.revision += 1
        rows[:, 1] = 0
        if "Tracks" in self.viewer.layers:
            self.viewer.layers["Tracks"].data = rows
        elif len(rows):
            self.track_layer = self.viewer.add_tracks(rows, name="Tracks")
        if "Tracks" in self.viewer.layers:
            self.viewer.layers["Tracks"].tail_length = len(self.session.channel(dataset, channel)) * 2

    def _start_core_tracking(self, dataset, channel):
        identifier, is_sequence = self.session.binding(dataset, channel)
        model = self.session.dataset(dataset)
        settings = TrackingSettings(
            search_range=float(self.gui.trackpy_search_range.value()),
            memory=int(self.gui.trackpy_memory.value()),
            min_length=int(self.gui.min_track_length.value()),
            space=CoordinateSpace.RAW, frame_clock="channel",
        )
        self.worker = Worker(
            _track_model, model.snapshot(), identifier, is_sequence, channel,
            settings, self.gui.remove_unlinked.isChecked(), self.stop_event,
        )
        self.worker.signals.result.connect(partial(
            self._publish_tracks, dataset=dataset, channel=channel,
            expected=state_token(model), revision=self.session.revision,
        ))
        self.worker.signals.error.connect(lambda error: self.molseeq_notification(str(error[1])))
        self.worker.signals.finished.connect(self.tracking_finished)
        self.threadpool.start(self.worker)

    def initialise_tracking(self):

        try:

            dataset = self.gui.tracking_dataset.currentText()
            channel = self.gui.tracking_channel.currentText()

            _, n_locs, fitted = self.get_loc_dict(dataset, channel.lower(), type="localisations")

            if n_locs > 0 and fitted == True:

                self.update_ui(init=True)

                self._start_core_tracking(dataset, channel.lower())

            else:
                print(f"No localisations found for {dataset} - {channel}`")

        except:
            print(traceback.format_exc())
            self.update_ui()


