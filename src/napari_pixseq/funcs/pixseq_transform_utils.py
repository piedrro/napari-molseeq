import traceback
import numpy as np
import cv2
import os
from functools import partial
from napari_pixseq.funcs.pixseq_utils_compute import Worker
from qtpy.QtWidgets import QFileDialog
import math
import json
from datetime import datetime


def transform_image(img, transform_matrix,
        transform_mode = "homography", progress_callback=None):

    w, h = img.shape[-2:]

    n_frames = img.shape[0]
    n_segments = math.ceil(n_frames / 100)
    image_splits = np.array_split(img, n_segments)

    transformed_image = []

    iter = 0

    for index, image in enumerate(image_splits):

        image = np.moveaxis(image, 0, -1)

        if transform_mode == "homography":
            image = cv2.warpPerspective(image, transform_matrix, (h, w),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        elif transform_mode == "affine":
            image = cv2.warpAffine(image, transform_matrix, (h, w),
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        transformed_image.append(image)
        iter += 250
        progress = int((iter / n_frames) * 100)

        if progress_callback is not None:
            progress_callback(progress)

    transformed_image = np.dstack(transformed_image)
    transformed_image = np.moveaxis(transformed_image, -1, 0)

    return transformed_image


class _tranform_utils:

    def normalize_image(self, img, norm_method="minmax"):

        if norm_method == "minmax":
            img = img - np.min(img)
            img = img / np.max(img)
        elif norm_method == "mean":
            img = img - np.mean(img)
            img = img / np.std(img)

        return img


    def compute_transform_matrix(self):

        try:
            if self.dataset_dict != {}:

                dataset_name = self.gui.tform_compute_dataset.currentText()
                target_channel = self.gui.tform_compute_target_channel.currentText()
                reference_channel = self.gui.tform_compute_ref_channel.currentText()

                target_locs = None
                reference_locs = None

                if dataset_name in self.localisation_dict["fiducials"].keys():

                    fiducial_dict = self.localisation_dict["fiducials"][dataset_name]

                    if target_channel.lower() in fiducial_dict.keys():
                        target_locs = fiducial_dict[target_channel.lower()]["localisations"]

                    if reference_channel.lower() in fiducial_dict.keys():
                        reference_locs = fiducial_dict[reference_channel.lower()]["localisations"]

                if len(reference_locs) > 0 and len(target_locs) > 0:

                    reference_points = [[loc.x, loc.y] for loc in reference_locs]
                    target_points = [[loc.x, loc.y] for loc in target_locs]

                    reference_points = np.array(reference_points).astype(np.float32)
                    target_points = np.array(target_points).astype(np.float32)

                    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

                    matches = bf.match(reference_points, target_points)
                    matches = sorted(matches, key=lambda x: x.distance)

                    reference_points = np.float32([reference_points[m.queryIdx] for m in matches]).reshape(-1, 2)
                    target_points = np.float32([target_points[m.trainIdx] for m in matches]).reshape(-1, 2)

                    self.transform_matrix, _ = cv2.findHomography(target_points, reference_points, cv2.RANSAC)

                    print(f"Transform Matrix\n: {self.transform_matrix}")

                    if self.gui.save_tform.isChecked():
                        self.save_transform_matrix()

        except:
            print(traceback.format_exc())
            pass


    def save_transform_matrix(self):

        try:

            if self.transform_matrix is not None:

                # get save file name and path
                date = datetime.now().strftime("%y%m%d")
                file_name = f'pixseq_transform_matrix-{date}.txt'

                dataset_name = self.gui.tform_compute_dataset.currentText()
                channel_name = self.gui.tform_compute_target_channel.currentText()

                path = self.dataset_dict[dataset_name][channel_name.lower()]["path"]
                path_directory = os.path.dirname(path)

                tform_path = os.path.join(path_directory, file_name)

                tform_path = QFileDialog.getSaveFileName(self, 'Save transform matrix', tform_path, 'Text files (*.txt)')[0]

                if tform_path != "":
                    self.pixseq_notification(f"Saving transform matrix to {tform_path}")

                    with open(tform_path, 'w') as filehandle:
                        json.dump(self.transform_matrix.tolist(), filehandle)

        except:
            print(traceback.format_exc())
            pass


    def _apply_transform_matrix_finished(self):

        try:

            self.image_layer.data = self.dataset_dict[self.active_dataset][self.active_channel]["data"]

            self.update_ui()

        except:
            print(traceback.format_exc())
            pass

    def _apply_transform_matrix(self, progress_callback=None):

        self.pixseq_notification("Applying transform matrix...")

        try:

            if self.dataset_dict != {}:

                from qtpy.QtWidgets import QComboBox
                self.gui.tform_apply_target = self.gui.self.gui.tform_apply_target

                apply_channel = self.gui.tform_apply_target.currentText()

                if "donor" in apply_channel.lower():
                    ref_emission = "d"
                else:
                    ref_emission = "a"

                target_images = []
                total_frames = 0
                iter = 0

                for dataset_name, dataset_dict in self.dataset_dict.items():
                    for channel_name, channel_dict in dataset_dict.items():
                        channel_ref = channel_dict["channel_ref"]
                        channel_emission = channel_ref[-1].lower()
                        if channel_emission == ref_emission:
                            n_frames = channel_dict["data"].shape[0]
                            total_frames += n_frames
                            target_images.append({"dataset_name": dataset_name,"channel_name": channel_name})

                for i in range(len(target_images)):

                    dataset_name = target_images[i]["dataset_name"]
                    channel_name = target_images[i]["channel_name"]

                    img = self.dataset_dict[dataset_name][channel_name.lower()]["data"].copy()

                    def transform_progress(progress):
                        nonlocal iter
                        iter += progress
                        progress = int((iter / total_frames) * 100)
                        progress_callback.emit(progress)

                    img = transform_image(img, self.transform_matrix,progress_callback=transform_progress)
                    self.dataset_dict[dataset_name][channel_name.lower()]["data"] = img.copy()

        except:
            print(traceback.format_exc())
            pass


    def apply_transform_matrix(self):

        try:

            if self.dataset_dict != {}:

                if hasattr(self, "transform_matrix") == False:

                    self.pixseq_notification("No transform matrix loaded.")

                else:

                    if self.transform_matrix is None:

                        self.pixseq_notification("No transform matrix loaded.")

                    else:

                        self.update_ui(init=True)

                        self.worker = Worker(self._apply_transform_matrix)
                        self.worker.signals.progress.connect(partial(self.pixseq_progress, progress_bar=self.gui.tform_apply_progressbar))
                        self.worker.signals.finished.connect(self._apply_transform_matrix_finished)
                        self.worker.signals.error.connect(self.update_ui)
                        self.threadpool.start(self.worker)

        except:
            self.update_ui()

            print(traceback.format_exc())
            pass


    def import_transform_matrix(self):

        try:

            desktop = os.path.expanduser("~/Desktop")
            path, filter = QFileDialog.getOpenFileName(self, "Open Files", desktop, "Files (*.txt)")

            self.transform_matrix = None

            if path != "":
                if os.path.isfile(path) == True:
                    if path.endswith(".txt"):
                        with open(path, 'r') as f:
                            transform_matrix = json.load(f)

                    transform_matrix = np.array(transform_matrix, dtype=np.float64)

                    if transform_matrix.shape == (3, 3):
                        self.transform_matrix = transform_matrix

                        print(f"Loaded transformation matrix:\n{transform_matrix}")

                    else:
                        print("Transformation matrix is wrong shape, should be (3,3)")

        except:
            print(traceback.format_exc())
            pass





