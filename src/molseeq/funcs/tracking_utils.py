import traceback
import pandas as pd
import trackpy as tp
import numpy as np
from functools import partial

from molcore import CoordinateSpace, StaleResultError, TrackingSettings, tracking
from molcore.staging import adopt_dataset, state_token, transact
from molseeq.funcs.utils_compute import Worker


def _track_model(model, identifier, is_sequence, channel, settings, remove_unlinked,
                 cancel_event, progress_callback):
    options = {
        "settings": settings,
        "cancel_event": cancel_event,
        "progress": lambda update: progress_callback.emit(int(update.percent)),
        "num_cpus": 1,
    }
    if is_sequence:
        tracking.track_sequence(model, identifier, channel, **options)
        parts = model.sequences[identifier].acquisitions
        rows = model.tracks.query(
            sequence_id=identifier, acquisition_id=[item.id for item in parts],
            channel=channel, space="raw",
        )
    else:
        tracking.track(model, identifier, channel, **options)
        parts = [model.acquisitions[identifier]]
        rows = model.tracks.query(acquisition_id=identifier, channel=channel, space="raw")

    frame = np.empty(len(rows["id"]), dtype=np.int64)
    offset = 0
    for acquisition in parts:
        view = acquisition.channels[channel]
        mask = rows["acquisition_id"] == acquisition.id
        frame[mask] = view.channel_indices(rows["frame"][mask]) + offset
        offset += len(view)
    if remove_unlinked:
        ids = model.localizations.query(
            acquisition_id=[item.id for item in parts], channel=channel, space="raw"
        )["id"]
        model.localizations.delete(np.setdiff1d(ids, rows["id"]))
    _, labels = np.unique(rows["track_id"], return_inverse=True)
    return model, np.column_stack((labels + 1, frame, rows["y"], rows["x"]))


class _tracking_utils:

    def run_tracking(self, locs, progress_callback=None):

        tracks_array = []

        try:

            search_range = float(self.gui.trackpy_search_range.value())
            memory = int(self.gui.trackpy_memory.value())
            min_track_length = int(self.gui.min_track_length.value())

            columns = list(locs.dtype.names)

            locdf = pd.DataFrame(locs, columns=columns)

            tracks_df = tp.link(locdf, search_range=search_range, memory=memory)

            # Count the frames per track
            track_lengths = tracks_df.groupby('particle').size()

            # Filter tracks by length
            valid_tracks = track_lengths[track_lengths >= min_track_length].index
            tracks_df = tracks_df[tracks_df['particle'].isin(valid_tracks)]

            self.tracks = tracks_df

            track_index = 1
            for particle, group in tracks_df.groupby("particle"):
                tracks_df.loc[group.index, "particle"] = track_index
                track_index += 1

            tracks_df = tracks_df[['particle', 'frame', 'y', 'x']]
            tracks_df = tracks_df.sort_values(by=["particle", "frame"])

            tracks_array = tracks_df.to_records(index=False)
            tracks_array = [list(track) for track in tracks_array]
            tracks_array = np.array(tracks_array)

        except:
            print(traceback.format_exc())

        return tracks_array

    def process_tracking_results(self, tracks_array):

        try:

            dataset = self.gui.tracking_dataset.currentText()
            channel = self.gui.tracking_channel.currentText()
            remove_unlinked = self.gui.remove_unlinked.isChecked()

            n_frames = self.dataset_dict[dataset][channel.lower()]["data"].shape[0]

            layers_names = [layer.name for layer in self.viewer.layers]

            render_tracks = tracks_array.copy()
            render_tracks[:,1] = 0

            if "Tracks" not in layers_names:
                self.track_layer = self.viewer.add_tracks(render_tracks, name="Tracks")
            else:
                self.track_layer.data = render_tracks

            self.track_layer.tail_length = n_frames * 2

            if remove_unlinked:

                loc_dict = self.localisation_dict["localisations"][dataset][channel.lower()]

                locs = loc_dict["localisations"]
                n_locs = len(locs)

                filtered_locs = pd.DataFrame(tracks_array,
                    columns=["particle", "frame", "y", "x"])
                filtered_locs = filtered_locs.to_records(index=False)

                n_filtered = len(filtered_locs)

                n_removed = n_locs - n_filtered

                if n_removed > 0:
                    print(f"Removed {n_removed} unlinked localisations")

                    loc_dict["localisations"] = filtered_locs



        except:
            print(traceback.format_exc())


    def tracking_finished(self):

        self.draw_localisations()
        self.update_ui()

    def _publish_tracks(self, result, dataset, channel, expected, revision):
        if self.stop_event.is_set() or self.session.revision != revision:
            return
        snapshot, rows = result
        model = self.session.dataset(dataset)
        try:
            transact(model, lambda staged: adopt_dataset(staged, snapshot), expected=expected)
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
        if is_sequence and not hasattr(tracking, "track_sequence"):
            return False
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
        return True

    def initialise_tracking(self):

        try:

            dataset = self.gui.tracking_dataset.currentText()
            channel = self.gui.tracking_channel.currentText()

            loc_dict, n_locs, fitted = self.get_loc_dict(dataset, channel.lower(), type="localisations")

            if n_locs > 0 and fitted == True:

                self.update_ui(init=True)

                if self._start_core_tracking(dataset, channel.lower()):
                    return

                locs = loc_dict["localisations"].copy()

                worker = Worker(self.run_tracking, locs)
                worker.signals.result.connect(self.process_tracking_results)
                worker.signals.finished.connect(self.tracking_finished)
                self.threadpool.start(worker)

            else:
                print(f"No localisations found for {dataset} - {channel}`")

        except:
            print(traceback.format_exc())
            self.update_ui()


