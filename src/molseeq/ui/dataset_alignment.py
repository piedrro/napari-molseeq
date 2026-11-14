from concurrent.futures import CancelledError

import cv2
import numpy as np
from molcore.alignment import fit_transform
from scipy.optimize import least_squares

from molseeq.ui.channel_transforms import match_fiducials, warp_corrections


def compute_dataset_alignment(
    sources, align_dict, reference_dataset, model, cancel_event, progress_callback=None,
):
    if reference_dataset not in align_dict:
        raise ValueError(f"Missing fitted localisations for {reference_dataset}")

    transforms = {}
    for dataset in sources:
        if cancel_event.is_set():
            raise CancelledError()
        if dataset not in align_dict:
            raise ValueError(f"Missing fitted localisations for {dataset}")
        try:
            source, reference = match_fiducials(align_dict[dataset], align_dict[reference_dataset])
            fitted_model = model
            if fitted_model == "auto":
                if len(source) <= 2:
                    fitted_model = "translation"
                    source, reference = source[:1], reference[:1]
                elif len(source) == 3:
                    fitted_model = "affine"
                else:
                    fitted_model = "homography"
            matrix, _, _ = fit_transform(source, reference, model=fitted_model, ransac_threshold=5.0)
        except (ValueError, cv2.error) as error:
            raise ValueError(f"Could not align {dataset}: {error}") from error
        mode = "homography" if fitted_model == "homography" else "affine"
        operation = {
            "method": "dataset_alignment",
            "matrix": matrix.tolist(),
            "model": fitted_model,
            "reference_dataset": reference_dataset,
            "interpolation": "opencv_linear",
            "border_value": 0,
        }
        transforms[dataset] = matrix, mode, operation

    return warp_corrections(sources, transforms, cancel_event, progress_callback)


class DatasetAlignmentMixin:

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


    def align_datasets(self):
        datasets = self.dataset_dict
        if len(datasets) < 2:
            return

        reference = self.gui.align_reference_dataset.currentText()
        selected_channel = self.gui.align_reference_channel.currentText()
        if "Donor Channels" in selected_channel:
            mode = "Donor"
            target_channels = {"donor", "dd", "ad"}
        else:
            mode = "Acceptor"
            target_channels = {"acceptor", "aa", "da"}

        align_dict = {}
        missing = []
        localisations = self.localisation_dict["localisations"]
        for dataset, channels in datasets.items():
            selected = [
                localisations[dataset][channel]
                for channel in channels if channel in target_channels
            ]
            if not selected or any(not values.get("fitted") for values in selected):
                missing.append(dataset)
            else:
                align_dict[dataset] = selected[-1]["localisations"]

        if missing:
            self.molseeq_notification(f"Missing fitted {mode} localisations for {', '.join(missing)}")
            return
        if reference not in align_dict:
            self.molseeq_notification("Select a reference dataset for alignment.")
            return

        sources = {
            dataset: {channel: self.session.working_source(dataset, channel) for channel in channels}
            for dataset, channels in datasets.items() if dataset != reference
        }
        self._start_image_correction(
            compute_dataset_alignment, datasets, self.gui.align_progressbar,
            sources, align_dict, reference, self.gui.align_model.currentText().lower(),
        )
