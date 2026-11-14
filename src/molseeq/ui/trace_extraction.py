import copy
import traceback
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from molcore import StaleResultError
from molcore.staging import experiment_state_token
from numba.core.errors import NumbaPendingDeprecationWarning

from molseeq.trace_views import (
    compute_fixed_traces,
    publish_traces,
    trace_view,
)
from molseeq.ui.worker import Worker

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)
np.seterr(divide='ignore', invalid='ignore')

def crop_spot_data(image_shape, spot_bounds, spot_mask, background_mask=None):

    try:
        x1, x2, y1, y2 = spot_bounds
        crop = [0, spot_mask.shape[1], 0, spot_mask.shape[0]]

        if x1 < 0:
            crop[0] = abs(x1)
            x1 = 0
        if x2 > image_shape[1]:
            crop[1] = spot_mask.shape[1] - (x2 - image_shape[1])
            x2 = image_shape[1]
        if y1 < 0:
            crop[2] = abs(y1)
            y1 = 0
        if y2 > image_shape[0]:
            crop[3] = spot_mask.shape[0] - (y2 - image_shape[0])
            y2 = image_shape[0]

        corrected_bounds = [x1, x2, y1, y2]

        if spot_mask is not None:
            loc_mask = spot_mask.copy()
            loc_mask = loc_mask[crop[2]:crop[3], crop[0]:crop[1]]
        else:
            loc_mask = None

        if background_mask is not None:
            loc_bg_mask = background_mask.copy()
            loc_bg_mask = loc_bg_mask[crop[2]:crop[3], crop[0]:crop[1]]
        else:
            loc_bg_mask = None

    except:
        loc_mask = spot_mask
        loc_bg_mask = background_mask
        corrected_bounds = spot_bounds
        print(traceback.format_exc())

    return corrected_bounds, loc_mask, loc_bg_mask





class TraceExtractionMixin:

    def generate_spot_bounds(self, locs, box_size):

        spot_bounds = []

        for loc_index, loc in enumerate(locs):

            x,y = loc.x, loc.y

            if box_size % 2 == 0:
                x += 0.5
                y += 0.5
                x, y = round(x), round(y)
                x1 = x - (box_size // 2)
                x2 = x + (box_size // 2)
                y1 = y - (box_size // 2)
                y2 = y + (box_size // 2)
            else:
                # Odd spot width
                x, y = round(x), round(y)
                x1 = x - (box_size // 2)
                x2 = x + (box_size // 2)+1
                y1 = y - (box_size // 2)
                y2 = y + (box_size // 2)+1

            spot_bounds.append([x1,x2,y1,y2])

        return spot_bounds

    def _molseeq_compute_traces_finished(self):

        self.gui.compute_traces.setEnabled(True)

        self.populate_plot_combos()
        self.populate_export_combos()
        self.initialize_plot()

        self.update_ui()

    def generate_localisation_mask(self, spot_size, spot_shape="square", buffer_size=0, bg_width=1, plot=False):

        box_size = spot_size + (bg_width * 2) + (buffer_size * 2)

        # Create a grid of coordinates
        y, x = np.ogrid[:box_size, :box_size]

        # Adjust center based on box size
        if box_size % 2 == 0:
            center = (box_size / 2 - 0.5, box_size / 2 - 0.5)
        else:
            center = (box_size // 2, box_size // 2)

        if spot_shape.lower() == "circle":
            # Calculate distance from the center for circular mask
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

            # Central spot mask
            inner_radius = spot_size // 2
            mask = distance <= inner_radius

            # Buffer mask
            buffer_outer_radius = inner_radius + buffer_size
            buffer_mask = (distance > inner_radius) & (distance <= buffer_outer_radius)

            # Background mask (outside the buffer zone)
            background_outer_radius = buffer_outer_radius + bg_width
            background_mask = (distance > buffer_outer_radius) & (distance <= background_outer_radius)

        elif spot_shape.lower() == "square":
            # Create square mask
            half_size = spot_size // 2
            mask = (abs(x - center[0]) <= half_size) & (abs(y - center[1]) <= half_size)

            # Create square background mask (one pixel larger on each side)
            buffer_mask = (abs(x - center[0]) <= half_size + buffer_size) & (abs(y - center[1]) <= half_size + buffer_size)
            background_mask = (abs(x - center[0]) <= half_size + buffer_size + bg_width) & (abs(y - center[1]) <= half_size + buffer_size + bg_width)
            background_mask = background_mask & ~buffer_mask

        if plot == True:
            plt.figure(figsize=(6, 6))
            plt.imshow(mask, cmap='gray', interpolation='none')
            plt.xticks(np.arange(-0.5, box_size, 1), [])
            plt.yticks(np.arange(-0.5, box_size, 1), [])
            plt.grid(color='blue', linestyle='-', linewidth=2)
            plt.title(f"{box_size}x{box_size} Spot Mask")
            plt.show()

            plt.figure(figsize=(6, 6))
            plt.imshow(background_mask, cmap='gray', interpolation='none')
            plt.xticks(np.arange(-0.5, box_size, 1), [])
            plt.yticks(np.arange(-0.5, box_size, 1), [])
            plt.grid(color='blue', linestyle='-', linewidth=2)
            plt.title(f"{box_size}x{box_size} Background Mask")
            plt.show()

        return mask, buffer_mask, background_mask

    def generate_background_overlap_mask(self, locs, spot_mask, spot_background_mask, image_mask_shape):

        global_spot_mask = np.zeros(image_mask_shape, dtype=np.uint8)
        global_background_mask = np.zeros(image_mask_shape, dtype=np.uint8)

        spot_mask = spot_mask.astype(np.uint16)
        spot_background_mask = spot_background_mask.astype(np.uint16)

        spot_bounds = self.generate_spot_bounds(locs,  len(spot_mask[0]))

        for loc_index, bounds in enumerate(spot_bounds):

            [x1, x2, y1, y2], loc_mask, log_bg_mask = crop_spot_data(image_mask_shape,
                bounds, spot_mask,spot_background_mask)

            global_spot_mask[y1:y2, x1:x2] += loc_mask
            global_background_mask[y1:y2, x1:x2] += log_bg_mask

        global_spot_mask[global_spot_mask > 0] = 1
        global_background_mask[global_background_mask > 0] = 1

        intersection_mask = global_spot_mask & global_background_mask

        global_background_mask = global_background_mask - intersection_mask

        return global_background_mask, global_spot_mask

    def find_bleach_indices(self, data, background_data, mode="last", smooth=True, smooth_window=10, n_frames=10):
        def moving_average(data, window_size):
            cumsum = np.cumsum(np.insert(data, 0, 0))
            return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

        if smooth:
            data = moving_average(data, window_size=smooth_window)
            background_data = moving_average(background_data, window_size=smooth_window)

        def find_first_consistent_index(comparison):
            longest_true_seq = 0
            current_seq = 0
            last_false_index = -1

            for i, val in enumerate(comparison):
                if val:
                    current_seq += 1
                else:
                    if current_seq > longest_true_seq:
                        longest_true_seq = current_seq
                        last_false_index = i
                    current_seq = 0

            # Check the last sequence
            if current_seq > longest_true_seq:
                last_false_index = len(comparison)

            return last_false_index if last_false_index != -1 else -1

        def find_first_index(comparison):
            consecutive_frames = 0
            for i, val in enumerate(comparison):
                if val:
                    consecutive_frames += 1
                    if consecutive_frames >= n_frames:
                        return i - n_frames + 1
                else:
                    consecutive_frames = 0
            return -1

        # Comparison with background data
        comparison_bg = data < background_data
        index_bg = find_first_consistent_index(comparison_bg) if mode == "last" else find_first_index(comparison_bg)

        # # Comparison with zero
        # comparison_zero = data < 0
        # index_zero = find_first_consistent_index(comparison_zero) if mode == "last" else find_first_index(comparison_zero)

        # Return the smaller index of the two
        return index_bg

    def compute_photo_bleaching(self, spot_metric="spot_mean", background_metric="spot_mean_local_bg",
            mode="first", n_frames=10):

        try:

            for dataset_name, dataset_dict in self.traces_dict.items():

                channel_list = list(dataset_dict.keys())

                donor_channel = None
                acceptor_channel = None

                if set(["dd", "da", "ad", "aa"]).issubset(channel_list):
                    donor_channel = "dd"
                    acceptor_channel = "aa"
                elif set(["donor", "acceptor"]).issubset(channel_list):
                    donor_channel = "donor"
                    acceptor_channel = "acceptor"

                if donor_channel is not None and acceptor_channel is not None:

                    for spot_index in dataset_dict[donor_channel].keys():

                        donor_data = dataset_dict[donor_channel][spot_index][spot_metric].copy()
                        donor_background_data = dataset_dict[donor_channel][spot_index][background_metric].copy()
                        acceptor_data = dataset_dict[acceptor_channel][spot_index][spot_metric].copy()
                        acceptor_background_data = dataset_dict[acceptor_channel][spot_index][background_metric].copy()

                        donor_bleach_index = self.find_bleach_indices(donor_data, donor_background_data, mode=mode)
                        acceptor_bleach_index = self.find_bleach_indices(acceptor_data, acceptor_background_data, mode=mode)

                        if donor_bleach_index != -1 and acceptor_bleach_index != -1:
                            bleach_index = max(donor_bleach_index, acceptor_bleach_index)
                        else:
                            bleach_index= -1


                        for channel in dataset_dict.keys():
                            self.traces_dict[dataset_name][channel][spot_index]["bleach_index"] = bleach_index
                            self.traces_dict[dataset_name][channel][spot_index]["donor_bleach_index"] = donor_bleach_index
                            self.traces_dict[dataset_name][channel][spot_index]["acceptor_bleach_index"] = acceptor_bleach_index
                else:

                    for spot_index in dataset_dict[channel_list[0]].keys():

                        data = dataset_dict[channel_list[0]][spot_index][spot_metric].copy()
                        background_data = dataset_dict[channel_list[0]][spot_index][background_metric].copy()

                        bleach_index = self.find_bleach_indices(data, background_data, mode=mode)

                        for channel in dataset_dict.keys():
                            self.traces_dict[dataset_name][channel][spot_index]["bleach_index"] = bleach_index
                            self.traces_dict[dataset_name][channel][spot_index]["donor_bleach_index"] = bleach_index
                            self.traces_dict[dataset_name][channel][spot_index]["acceptor_bleach_index"] = bleach_index


        except:
            print(traceback.format_exc())
            pass


    def _molseeq_compute_traces(self, snapshot, settings, locs, box_size, expected, progress_callback):
        return compute_fixed_traces(
            snapshot, locs, settings, box_size=box_size, expected=expected,
            progress=progress_callback.emit, cancel_event=self.stop_event,
        )

    def _publish_trace_metrics(self, traces, revision, site_snapshot):
        if (
            not traces or self.stop_event.is_set() or self.session.revision != revision
            or self.bounding_box_token() != site_snapshot
        ):
            return
        try:
            publish_traces(self.session, traces)
            self.traces_dict = trace_view(self.session)
            self.compute_photo_bleaching()
        except (ValueError, KeyError, StaleResultError) as error:
            self.molseeq_notification(str(error))

    def molseeq_compute_traces(self):
        boxes = self.localisation_dict["bounding_boxes"]
        locs = boxes.get("localisations")
        if locs is None or not len(locs):
            self.molseeq_notification("Bounding Boxes required for trace computation.")
            return

        settings = (
            int(self.gui.traces_spot_size.currentText()),
            self.gui.traces_spot_shape.currentText(),
            int(self.gui.traces_background_buffer.currentText()),
            int(self.gui.traces_background_width.currentText()),
            self.gui.compute_global_background.isChecked(),
            self.gui.compute_with_picasso.isChecked(),
        )
        self.molseeq_notification(f"Computing traces for {len(locs)} bounding boxes.")
        self.update_ui(init=True)
        self.worker = Worker(
            self._molseeq_compute_traces, self.session.image_snapshot(),
            settings, locs.copy(), boxes["box_size"],
            experiment_state_token(self.session.experiment),
        )
        self.worker.signals.progress.connect(self.gui.compute_traces_progressbar.setValue)
        self.worker.signals.result.connect(
            partial(
                self._publish_trace_metrics,
                revision=self.session.revision,
                site_snapshot=self.bounding_box_token(),
            )
        )
        self.worker.signals.finished.connect(self._molseeq_compute_traces_finished)
        self.worker.signals.error.connect(lambda error: self.molseeq_notification(str(error[1])))
        self.threadpool.start(self.worker)

    def visualise_background_masks(self):

        self.gui.compute_traces.setEnabled(True)

        try:


            if "bounding_boxes" in self.localisation_dict.keys():
                if "localisations" in self.localisation_dict["bounding_boxes"].keys():

                    spot_size = int(self.gui.traces_spot_size.currentText())
                    spot_shape = self.gui.traces_spot_shape.currentText()
                    buffer_size = int(self.gui.traces_background_buffer.currentText())
                    bg_width = int(self.gui.traces_background_width.currentText())

                    localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])
                    locs = localisation_dict["localisations"]

                    spot_mask, buffer_mask, bg_mask = self.generate_localisation_mask(spot_size, spot_shape, buffer_size, bg_width)

                    mask_shape = self.dataset_dict[self.active_dataset][self.active_channel]["data"].shape[-2:]

                    background_overlap_mask, _ = self.generate_background_overlap_mask(locs, buffer_mask, bg_mask, mask_shape)

                    spot_bounds = self.generate_spot_bounds(locs, len(bg_mask[0]))

                    bg_mask = bg_mask.astype(np.uint8)

                    mask_shape = self.dataset_dict[self.active_dataset][self.active_channel]["data"].shape[-2:]

                    global_spot_mask = np.zeros(mask_shape, dtype=np.uint16)

                    for loc_index, bounds in enumerate(spot_bounds):

                        [x1, x2, y1, y2], loc_bg_mask, _ = crop_spot_data(mask_shape, bounds, bg_mask)

                        temp_mask = np.zeros(mask_shape, dtype=np.uint8)
                        temp_mask[y1:y2, x1:x2] += loc_bg_mask

                        global_spot_mask[temp_mask > 0] = loc_index + 1

                    binary_spot_mask = global_spot_mask > 0
                    overlap_mask = binary_spot_mask & background_overlap_mask
                    inverse_overlap_mask = np.logical_not(overlap_mask)

                    global_spot_mask[inverse_overlap_mask] = 0

                    if "Background Mask" in self.viewer.layers:
                        self.viewer.layers.remove("Background Mask")
                    self.viewer.add_labels(global_spot_mask,
                        opacity=0.4, name="Background Mask")

        except:
            print(traceback.format_exc())

    def visualise_spot_masks(self):

        try:

            if "bounding_boxes" in self.localisation_dict.keys():
                if "fitted" in self.localisation_dict["bounding_boxes"].keys():

                    spot_size = int(self.gui.traces_spot_size.currentText())
                    spot_shape = self.gui.traces_spot_shape.currentText()
                    buffer_size = int(self.gui.traces_background_buffer.currentText())
                    bg_width = int(self.gui.traces_background_width.currentText())

                    localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])
                    locs = localisation_dict["localisations"]

                    spot_mask, buffer_mask, spot_background_mask = self.generate_localisation_mask(spot_size, spot_shape, buffer_size, bg_width)

                    spot_bounds = self.generate_spot_bounds(locs, len(spot_mask[0]))

                    spot_mask = spot_mask.astype(np.uint8)

                    mask_shape = self.dataset_dict[self.active_dataset][self.active_channel]["data"].shape[-2:]

                    global_spot_mask = np.zeros(mask_shape, dtype=np.uint16)

                    for loc_index, bounds in enumerate(spot_bounds):

                        [x1, x2, y1, y2], loc_mask, _ = crop_spot_data(mask_shape, bounds, spot_mask)

                        temp_mask = np.zeros(mask_shape, dtype=np.uint8)
                        temp_mask[y1:y2, x1:x2] += loc_mask
                        global_spot_mask[temp_mask > 0] = loc_index + 1

                    if "Spot Mask" in self.viewer.layers:
                        self.viewer.layers.remove("Spot Mask")
                    self.viewer.add_labels(
                        global_spot_mask,
                        opacity=0.8,
                        name="Spot Mask")

        except:
            print(traceback.format_exc())
            pass




camera_info = {"Baseline": 100.0, "Gain": 1, "Sensitivity": 1.0, "qe": 0.9, }

# method = "lq"
# gain = 1
#
# total_locs = 0
# progress_dict = {}
