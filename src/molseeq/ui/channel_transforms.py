import json
import os
import traceback
from concurrent.futures import CancelledError
from datetime import datetime
from functools import partial

import cv2
import numpy as np
from molcore import StaleResultError
from molcore.alignment import fit_transform
from molcore.staging import state_token
from qtpy.QtWidgets import QFileDialog

from molseeq.session import ImageCorrection
from molseeq.ui.worker import Worker


def match_fiducials(source_locs, reference_locs):
    """Pair nearby source and reference localisations using the existing cross-check."""
    source = np.array([(loc.x, loc.y) for loc in source_locs], dtype=np.float32).reshape(-1, 2)
    reference = np.array([(loc.x, loc.y) for loc in reference_locs], dtype=np.float32).reshape(-1, 2)
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = sorted(matcher.match(reference, source), key=lambda match: match.distance)
    return source[[m.trainIdx for m in matches]], reference[[m.queryIdx for m in matches]]


def transform_image(
    img, transform_matrix, transform_mode="homography",
    progress_callback=None, cancel_event=None,
):
    height, width = img.shape[-2:]
    n_frames = img.shape[0]
    result = np.empty(img.shape, dtype=img.dtype)
    for begin in range(0, n_frames, 100):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        end = min(begin + 100, n_frames)
        chunk = np.moveaxis(img[begin:end], 0, -1)
        if transform_mode == "homography":
            warped = cv2.warpPerspective(
                chunk, transform_matrix, (width, height),
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        else:
            warped = cv2.warpAffine(
                chunk, transform_matrix[:2], (width, height),
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        if warped.ndim == 2:
            warped = warped[..., None]
        result[begin:end] = np.moveaxis(warped, -1, 0)
        if progress_callback is not None:
            progress_callback(100 * end / n_frames)
    return result


def warp_corrections(sources, transforms, cancel_event, progress_callback):
    """Apply captured transforms to detached images and return staged corrections."""
    total_frames = sum(source.shape[0] for channels in sources.values() for source in channels.values())
    completed = 0
    corrections = {}
    for dataset, channels in sources.items():
        matrix, mode, operation = transforms[dataset]
        images = {}
        for channel, source in channels.items():
            def progress(percent):
                if progress_callback is not None:
                    done = completed + source.shape[0] * percent / 100
                    progress_callback.emit(int(100 * done / total_frames))

            images[channel] = transform_image(
                source, matrix, mode, progress_callback=progress, cancel_event=cancel_event
            )
            completed += source.shape[0]
        corrections[dataset] = ImageCorrection(images, [], operation)
    if cancel_event.is_set():
        raise CancelledError()
    return corrections


def compute_channel_transform(sources, matrix, model, cancel_event, progress_callback=None):
    operation = {
        "method": "channel_transform",
        "matrix": matrix.tolist(),
        "interpolation": "opencv_linear",
        "border_value": 0,
    }
    if model is not None:
        operation["model"] = model
    transforms = {dataset: (matrix, "homography", operation) for dataset in sources}
    return warp_corrections(sources, transforms, cancel_event, progress_callback)


class ChannelTransformsMixin:

    def normalize_image(self, img, norm_method="minmax"):

        if norm_method == "minmax":
            img = img - np.min(img)
            img = img / np.max(img)
        elif norm_method == "mean":
            img = img - np.mean(img)
            img = img / np.std(img)

        return img


    def compute_transform_matrix(self):
        self.transform_matrix = None
        self.transform_model = None
        if not self.dataset_dict:
            return

        dataset = self.gui.tform_compute_dataset.currentText()
        source_channel = self.gui.tform_compute_target_channel.currentText().lower()
        reference_channel = self.gui.tform_compute_ref_channel.currentText().lower()
        model = self.gui.tform_compute_model.currentText().lower()
        fiducials = self.localisation_dict["localisations"].get(dataset, {})
        source_locs = fiducials.get(source_channel, {}).get("localisations", ())
        reference_locs = fiducials.get(reference_channel, {}).get("localisations", ())
        if not len(source_locs) or not len(reference_locs):
            self.molseeq_notification("Both channels need fitted localisations.")
            return

        try:
            source, reference = match_fiducials(source_locs, reference_locs)
            matrix, _, _ = fit_transform(source, reference, model=model, ransac_threshold=3.0)
        except (ValueError, cv2.error) as error:
            self.molseeq_notification(f"Could not fit channel alignment: {error}")
            return

        self.transform_matrix = matrix
        self.transform_model = model
        if self.gui.save_tform.isChecked():
            self.save_transform_matrix()


    def save_transform_matrix(self):

        try:

            if self.transform_matrix is not None:

                # get save file name and path
                date = datetime.now().strftime("%y%m%d")
                file_name = f'molseeq_transform_matrix-{date}.txt'

                dataset_name = self.gui.tform_compute_dataset.currentText()
                channel_name = self.gui.tform_compute_target_channel.currentText()

                path = self.dataset_dict[dataset_name][channel_name.lower()]["path"]
                path_directory = os.path.dirname(path)

                tform_path = os.path.join(path_directory, file_name)

                tform_path = QFileDialog.getSaveFileName(self, 'Save transform matrix', tform_path, 'Text files (*.txt)')[0]

                if tform_path != "":
                    self.molseeq_notification(f"Saving transform matrix to {tform_path}")

                    with open(tform_path, 'w') as filehandle:
                        json.dump(self.transform_matrix.tolist(), filehandle)

        except:
            print(traceback.format_exc())
            pass


    def _start_image_correction(self, compute, names, progress_bar, *args):
        revision = self.session.revision
        expected = {name: state_token(self.session.dataset(name)) for name in names}
        self.update_ui(init=True)
        self.worker = Worker(compute, *args, self.stop_event)
        self.worker.signals.progress.connect(
            partial(self.molseeq_progress, progress_bar=progress_bar)
        )
        self.worker.signals.result.connect(
            partial(self._publish_warp, revision=revision, expected=expected)
        )
        self.worker.signals.error.connect(self._report_warp_error)
        self.worker.signals.finished.connect(self.update_ui)
        self.threadpool.start(self.worker)

    def _report_warp_error(self, error):
        if not isinstance(error[1], CancelledError):
            self.molseeq_notification(str(error[1]))

    def _publish_warp(self, corrections, *, revision, expected):
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

    def apply_transform_matrix(self):
        if not self.dataset_dict:
            return
        if self.transform_matrix is None:
            self.molseeq_notification("No transform matrix loaded.")
            return

        target = self.gui.tform_apply_target.currentText().lower()
        emission = "d" if "donor" in target else "a"
        sources = {}
        for dataset, channels in self.dataset_dict.items():
            selected = {
                channel: self.session.working_source(dataset, channel)
                for channel, metadata in channels.items()
                if metadata["channel_ref"][-1].lower() == emission
            }
            if selected:
                sources[dataset] = selected
        if sources:
            self._start_image_correction(
                compute_channel_transform, sources, self.gui.tform_apply_progressbar,
                sources, self.transform_matrix.copy(), getattr(self, "transform_model", None),
            )

    def import_transform_matrix(self):

        try:

            desktop = os.path.expanduser("~/Desktop")
            path, filter = QFileDialog.getOpenFileName(self, "Open Files", desktop, "Files (*.txt)")

            self.transform_matrix = None
            self.transform_model = None

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





