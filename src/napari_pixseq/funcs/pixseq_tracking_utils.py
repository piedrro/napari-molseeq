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

                # remove all but first particle
                # particles = tracked['particle'].unique()
                # tracked = tracked[tracked['particle'] == particles[0]]

                tracked = tracked[['particle', 'frame', 'y', 'x']]

                tracks = []

                track_index = 1
                for particle, group in tracked.groupby("particle"):

                    group['particle'] = track_index
                    track = group.to_records(index=False)
                    track = [list(track) for track in track]
                    tracks.extend(track)
                    track_index += 1

                layers_names = [layer.name for layer in self.viewer.layers]

                if "Tracks" not in layers_names:
                    self.track_layer = self.viewer.add_tracks(tracks, name="Tracks")
                else:
                    self.track_layer.data = tracks








                #remove rows where there is only one value of particle

                # tracked = tracked[tracked.groupby('particle').particle.transform('count') > 1]
                #
                # for particle, group in tracked.groupby("particle"):
                #
                #     if len(group) == 1:
                #         print(f"Particle {particle} has only one frame")

            else:
                print(f"No fiducials found for {dataset} - {channel}`")

        except:
            print(traceback.format_exc())
            pass

