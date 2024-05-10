from sklearn.cluster import DBSCAN
import numpy as np
import traceback
from napari_pixseq.funcs.pixseq_utils_compute import Worker


class _cluster_utils:

    def _cluster_localisations_finished(self):

        try:
            mode = self.gui.cluster_mode.currentText()

            if "fiducials" in mode.lower():
                self.draw_fiducials(update_vis=True)
            else:
                self.draw_bounding_boxes()

            self.update_ui()

        except:
            print(traceback.format_exc())
            pass


    def remove_overlapping_coords(self, coordinates, min_distance):

        # Calculate all pairwise differences
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]

        # Calculate squared distances
        dist_squared = np.sum(diff ** 2, axis=-1)

        # Check if the array is of integer type
        if coordinates.dtype.kind in 'iu':
            # Use the maximum integer value for the diagonal if the array is of integer type
            max_int_value = np.iinfo(coordinates.dtype).max
            np.fill_diagonal(dist_squared, max_int_value)
        else:
            # Use infinity for the diagonal if the array is of float type
            np.fill_diagonal(dist_squared, np.inf)

        # Identify overlapping coordinates (distance less than X)
        overlapping = np.any(dist_squared < min_distance ** 2, axis=1)

        # Filter out overlapping coordinates
        filtered_coordinates = coordinates[~overlapping]

        return filtered_coordinates



    def _cluster_localisations(self, progress_callback=None, eps=0.1, min_samples=20):

        result = None, None, None

        try:

            mode = self.gui.cluster_mode.currentText()
            dataset = self.gui.cluster_dataset.currentText()
            channel = self.gui.cluster_channel.currentText()
            remove_overlapping = self.gui.dbscan_remove_overlapping.isChecked()

            loc_dict, n_locs, fitted = self.get_loc_dict(dataset, channel.lower(), type = "fiducials")

            locs = loc_dict["localisations"]
            box_size = loc_dict["box_size"]

            n_frames = len(np.unique([loc.frame for loc in locs]))

            cluster_dataset = np.vstack((locs.x, locs.y)).T

            # Applying DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            dbscan.fit(cluster_dataset)

            # Extracting labels
            labels = dbscan.labels_

            # Finding unique clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            unique_labels = set(labels)

            # Filtering out noise (-1 label)
            filtered_data = cluster_dataset[labels != -1]

            # Corresponding labels after filtering out noise
            filtered_labels = labels[labels != -1]

            # Finding cluster centers
            cluster_centers = np.array([filtered_data[filtered_labels == i].mean(axis=0) for i in range(n_clusters)])

            if remove_overlapping:
                cluster_centers = self.remove_overlapping_coords(cluster_centers,
                    min_distance = box_size)

            if "fiducials" not in mode.lower():
                n_frames = 1

            clustered_locs = []
            clustered_loc_centers = []
            render_locs = {}

            for cluster_index in range(len(cluster_centers)):
                for frame_index in range(n_frames):
                    [locX, locY] = cluster_centers[cluster_index].copy()
                    new_loc = (int(frame_index), float(locX), float(locY))
                    clustered_locs.append(new_loc)

                    if frame_index not in render_locs.keys():
                        render_locs[frame_index] = []

                    render_locs[frame_index].append([locY, locX])
                    clustered_loc_centers.append([frame_index, locY, locX])

            # Convert list to recarray
            clustered_locs = np.array(clustered_locs, dtype=[('frame', '<u4'), ('x', '<f4'), ('y', '<f4')]).view(np.recarray)

            result = (clustered_locs, clustered_loc_centers, render_locs)

        except:
            print(traceback.format_exc())
            result = None

        return result


    def _cluster_localisations_result(self, result):

        try:

            if result is not None:

                locs, loc_centers, render_locs = result

                mode = self.gui.cluster_mode.currentText()
                dataset = self.gui.cluster_dataset.currentText()
                channel = self.gui.cluster_channel.currentText()

                fiducial_dict = self.localisation_dict["fiducials"][dataset][channel.lower()]
                bbox_dict = self.localisation_dict["bounding_boxes"]

                if "fiducials" in mode.lower():

                    fiducial_dict = self.localisation_dict["fiducials"][dataset][channel.lower()]

                    fiducial_dict["localisations"] = locs
                    fiducial_dict["localisation_centres"] = loc_centers
                    fiducial_dict["render_locs"] = render_locs

                else:

                    if "box_size" in fiducial_dict.keys():
                        box_size = fiducial_dict["box_size"]
                    else:
                        box_size = int(self.gui.picasso_box_size.currentText())

                    bbox_dict["localisations"] = locs
                    bbox_dict["localisation_centres"] = loc_centers
                    bbox_dict["box_size"] = box_size

        except:
            print(traceback.format_exc())
            pass



    def check_number(self, string):

        if string.isdigit():
            number = int(string)
        elif string.replace('.', '', 1).isdigit() and string.count('.') < 2:
            number = float(string)
        else:
            number = None

        return number

    def pixseq_cluster_localisations(self):

        try:

            dataset = self.gui.cluster_dataset.currentText()
            channel = self.gui.cluster_channel.currentText()
            mode = self.gui.cluster_mode.currentText()
            eps = self.gui.cluster_eps.text()
            min_samples = self.gui.dbscan_min_samples.text()

            eps = self.check_number(eps)
            min_samples = self.check_number(min_samples)

            loc_dict, n_locs, fitted = self.get_loc_dict(dataset, channel.lower())

            if n_locs == 0 or fitted == False:
                self.pixseq_notification("Localisation clustering requires fitted fiducials.")

            if n_locs > 0 and fitted and eps is not None and min_samples is not None:

                self.update_ui(init = True)

                worker = Worker(self._cluster_localisations, eps=eps, min_samples=min_samples)
                worker.signals.result.connect(self._cluster_localisations_result)
                worker.signals.finished.connect(self._cluster_localisations_finished)
                worker.signals.error.connect(self.update_ui)
                self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.update_ui()
            pass