import traceback
import numpy as np
import cv2
from functools import partial
from napari_pixseq.funcs.pixseq_utils_compute import Worker
from napari_pixseq.funcs.pixseq_transform_utils import transform_image
from scipy.optimize import least_squares

class _align_utils:

    def update_align_reference_channel(self):

        try:

            acceptor_channels = []
            donor_channels = []

            self.align_channels_dict = {}

            for datast_name, dataset_dict in self.dataset_dict.items():
                for channel in self.dataset_dict[datast_name].keys():
                    if channel.lower() in ["donor","dd","ad"]:

                        if channel.lower() == "donor":
                            channel = channel.capitalize()
                        else:
                            channel = channel.upper()

                        donor_channels.append(channel)

                    else:

                        if channel.lower() == "acceptor":
                            channel = channel.capitalize()
                        else:
                            channel = channel.upper()

                        acceptor_channels.append(channel)

            acceptor_channels = np.unique(acceptor_channels).tolist()
            donor_channels = np.unique(donor_channels).tolist()

            acceptor_channels = "Acceptor Channels: [" + "/ ".join(acceptor_channels) + "]"
            donor_channels = "Donor Channels: [" + "/".join(donor_channels) + "]"

            self.gui.align_reference_channel.blockSignals(True)
            self.gui.align_reference_channel.clear()
            self.gui.align_reference_channel.addItem(acceptor_channels)
            self.gui.align_reference_channel.addItem(donor_channels)
            self.gui.align_reference_channel.blockSignals(False)

        except:
            pass

    def affine_transform_matrix(self, points_src, points_dst):
        # Function to optimize
        def min_func(params):
            a, b, c, d, e, f = params
            transformed = np.dot(points_src, np.array([[a, b], [c, d]])) + np.array([e, f])
            return np.ravel(transformed - points_dst)

        # Initial guess
        x0 = np.array([1, 0, 0, 1, 0, 0])

        # Solve using least squares
        result = least_squares(min_func, x0)

        # Construct the transformation matrix
        a, b, c, d, e, f = result.x
        matrix = np.array([[a, b, e], [c, d, f], [0, 0, 1]])

        return matrix


    def _align_datasets_finished(self):

        try:

            self.update_active_image()
            self.image_layer.data = self.dataset_dict[self.active_dataset][self.active_channel]["data"]

            self.update_ui()

        except:
            print(traceback.format_exc())
            pass

    def _align_datasets(self, progress_callback, align_dict):

        try:

            reference_dataset = self.gui.align_reference_dataset.currentText()
            reference_channel = self.gui.align_reference_channel.currentText()

            dataset_list = list(self.dataset_dict.keys())
            dataset_list.remove(reference_dataset)

            total_frames = 0
            for dataset in dataset_list:
                for channel_name, channel_dict in self.dataset_dict[dataset].items():
                    total_frames += channel_dict["data"].shape[0]

            dst_locs = align_dict[reference_dataset].copy()

            iter = 0

            for dataset in dataset_list:

                src_locs = align_dict[dataset].copy()

                dst_pts = [[loc.x, loc.y] for loc in dst_locs]
                src_pts = [[loc.x, loc.y] for loc in src_locs]

                dst_pts = np.array(dst_pts).astype(np.float32)
                src_pts = np.array(src_pts).astype(np.float32)

                bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

                matches = bf.match(dst_pts, src_pts)
                matches = sorted(matches, key=lambda x: x.distance)

                dst_pts = np.float32([dst_pts[m.queryIdx] for m in matches]).reshape(-1, 2)
                src_pts = np.float32([src_pts[m.trainIdx] for m in matches]).reshape(-1, 2)

                if len(dst_pts) > 0:

                    if len(dst_pts) == 1 or len(dst_pts) == 2:

                        dst_point = dst_pts[0]
                        src_point = src_pts[0]

                        # Calculate translation vector
                        tx = dst_point[0] - src_point[0]
                        ty = dst_point[1] - src_point[1]

                        # Translation matrix
                        transform_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                        transform_mode = "affine"

                    elif len(dst_pts) == 3:

                        transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)
                        transform_mode = "affine"

                    elif len(dst_pts) > 3:
                        transform_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        transform_mode = "homography"

                    if transform_matrix.shape in [(2, 3), (3, 3)]:

                        for channel_name, channel_dict in self.dataset_dict[dataset].items():

                            self.pixseq_notification(f"Aligning {dataset} {channel_name}...")

                            img = channel_dict["data"].copy()

                            def transform_progress(progress):
                                nonlocal iter
                                iter += progress
                                progress = int((iter / total_frames) * 100)
                                progress_callback.emit(progress)

                            img = transform_image(img, transform_matrix,
                                transform_mode = transform_mode,
                                progress_callback=transform_progress)

                            self.dataset_dict[dataset][channel_name.lower()]["data"] = img.copy()

        except:
            print(traceback.format_exc())
            pass


    def align_datasets(self):

        try:

            if self.dataset_dict != {}:

                align_dataset = self.gui.align_reference_dataset.currentText()
                align_channel = self.gui.align_reference_channel.currentText()

                if "Donor Channels" in align_channel:
                    channel_mode = "Donor"
                    target_channels = ["donor", "dd", "ad"]
                else:
                    channel_mode = "Acceptor"
                    target_channels = ["acceptor", "aa", "da"]

                missing_fiducial_list = []

                align_dict = {}

                for dataset_name in self.dataset_dict.keys():
                    if dataset_name not in self.localisation_dict["localisations"].keys():
                        missing_fiducial_list.append(dataset_name)
                    else:
                        dataset_channels = list(self.dataset_dict[dataset_name].keys())
                        reference_channels = [channel.lower() for channel in dataset_channels if channel in target_channels]

                        for channel in reference_channels:
                            if channel not in self.localisation_dict["localisations"][dataset_name].keys():
                                missing_fiducial_list.append(dataset_name)
                            else:
                                localisation_dict = self.localisation_dict["localisations"][dataset_name][channel]
                                if "fitted" not in localisation_dict.keys():
                                    missing_fiducial_list.append(dataset_name)
                                else:
                                    if localisation_dict["fitted"] == False:
                                        missing_fiducial_list.append(dataset_name)
                                    else:
                                        align_dict[dataset_name] = localisation_dict["localisations"]


                if len(missing_fiducial_list) > 0:
                    missing_fiducial_list = ", ".join(missing_fiducial_list)
                    self.pixseq_notification(f"Missing fitted {channel_mode} localisations for {missing_fiducial_list}")
                else:

                    self.update_ui(init=True)

                    self.worker = Worker(self._align_datasets, align_dict=align_dict)
                    self.worker.signals.progress.connect(partial(self.pixseq_progress, progress_bar=self.gui.align_progressbar))
                    self.worker.signals.finished.connect(self._align_datasets_finished)
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)

        except:
            print(traceback.format_exc())

            self.update_ui()
