import traceback
import pandas as pd
import trackpy as tp
import numpy as np

class _tracking_utils:

    def initialise_tracking(self):

        try:

            dataset = self.gui.tracking_dataset.currentText()
            channel = self.gui.tracking_channel.currentText()
            search_range = float(self.gui.trackpy_search_range.value())
            memory = int(self.gui.trackpy_memory.value())
            min_track_length = int(self.gui.min_track_length.value())
            remove_unlinked = self.gui.remove_unlinked.isChecked()

            n_frames = self.dataset_dict[dataset][channel.lower()]["data"].shape[0]

            loc_dict, n_locs, fitted = self.get_loc_dict(dataset, channel.lower(), type="fiducials")

            if n_locs > 0 and fitted == True:

                locs = loc_dict["localisations"]

                columns = list(locs.dtype.names)

                locdf = pd.DataFrame(locs, columns=columns)

                tracked = tp.link(locdf, search_range=search_range, memory=memory)

                # Count the frames per track
                track_lengths = tracked.groupby('particle').size()

                # Filter tracks by length
                valid_tracks = track_lengths[track_lengths >= min_track_length].index
                tracked = tracked[tracked['particle'].isin(valid_tracks)]

                self.tracks = tracked

                tracks = []

                track_index = 1
                for particle, group in tracked.groupby("particle"):

                    group['particle'] = track_index
                    group = group[['particle', 'frame', 'y', 'x']]
                    track = group.to_records(index=False)
                    track = [list(track) for track in track]
                    tracks.extend(track)
                    track_index += 1

                tracks = np.array(tracks)
                tracks[:, 1] = 0

                layers_names = [layer.name for layer in self.viewer.layers]

                if "Tracks" not in layers_names:
                    self.track_layer = self.viewer.add_tracks(tracks, name="Tracks")
                else:
                    self.track_layer.data = tracks

                self.track_layer.tail_length = n_frames*2

                if remove_unlinked:

                    filtered_locs = tracked.to_records(index=False)

                    n_filtered = len(filtered_locs)

                    n_removed = n_locs - n_filtered

                    if n_removed > 0:

                        print(f"Removed {n_removed} unlinked localisations")

                        render_locs = {}

                        for frame in np.unique(filtered_locs["frame"]):
                            frame_locs = filtered_locs[filtered_locs["frame"] == frame].copy()
                            render_locs[frame] = np.vstack((frame_locs.y, frame_locs.x)).T.tolist()

                        loc_dict = self.localisation_dict["fiducials"][dataset][channel.lower()]

                        loc_dict["localisations"] = filtered_locs
                        loc_dict["render_locs"] = render_locs

                        self.draw_fiducials(update_vis=True)


            else:
                print(f"No fiducials found for {dataset} - {channel}`")

        except:
            print(traceback.format_exc())
            pass

