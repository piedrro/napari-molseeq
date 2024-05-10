import copy
import numpy as np
import traceback
from napari_pixseq.funcs.pixseq_utils_compute import Worker
from functools import partial
import matplotlib.pyplot as plt
from qtpy.QtWidgets import QComboBox
import multiprocessing
from picasso.gaussmle import gaussmle
import warnings
from numba.core.errors import NumbaPendingDeprecationWarning
import time
import concurrent.futures
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=NumbaPendingDeprecationWarning)
np.seterr(divide='ignore', invalid='ignore')

LOCS_DTYPE = [
    ("frame", "u4"),
    ("x", "f4"),
    ("y", "f4"),
    ("photons", "f4"),
    ("sx", "f4"),
    ("sy", "f4"),
    ("bg", "f4"),
    ("lpx", "f4"),
    ("lpy", "f4"),
    ("net_gradient", "f4"),
    ("likelihood", "f4"),
    ("iterations", "i4"),
    # ("loc_index", "u4")
]

def locs_from_fits(identifications, theta, CRLBs, likelihoods, iterations, box):

    box_offset = int(box / 2)
    y = theta[:, 0] + identifications.y - box_offset
    x = theta[:, 1] + identifications.x - box_offset
    lpy = np.sqrt(CRLBs[:, 0])
    lpx = np.sqrt(CRLBs[:, 1])
    locs = np.rec.array(
        (
            identifications.frame,
            x,
            y,
            theta[:, 2],
            theta[:, 5],
            theta[:, 4],
            theta[:, 3],
            lpx,
            lpy,
            identifications.net_gradient,
            likelihoods,
            iterations,
            # identifications.loc_index,
        ),
        dtype=LOCS_DTYPE,
    )
    locs.sort(kind="mergesort", order="frame")
    return locs

def get_loc_from_fit(loc, theta, CRLBs, likelihoods, iterations, box):

    box_offset = int(box / 2)
    y = (theta[:, 0]
         + loc.y - box_offset)
    x = (theta[:, 1] +
         loc.x - box_offset)
    lpy = np.sqrt(CRLBs[:, 0])
    lpx = np.sqrt(CRLBs[:, 1])

    loc.x = x
    loc.y = y
    loc.photons = theta[:, 2]
    loc.sx = theta[:, 5]
    loc.sy = theta[:, 4]
    loc.bg = theta[:, 3]
    loc.lpx = lpx
    loc.lpy = lpy
    loc.net_gradient = loc.net_gradient
    loc.likelihood = likelihoods
    loc.iterations = iterations

    return loc

def create_frame_locs(loc, n_frames):

    frame_locs = []
    for frame_index in range(n_frames):
        frame_loc = copy.deepcopy(loc)
        frame_loc.frame = frame_index
        frame_locs.append(frame_loc)

    frame_locs = np.array(frame_locs, dtype=loc).view(np.recarray)

    return frame_locs

def cut_spots(movie, ids_frame, ids_x, ids_y, box):

    n_spots = len(ids_x)
    r = int(box / 2)
    spots = np.zeros((n_spots, box, box), dtype=movie.dtype)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        spots[id] = movie[frame, yc - r : yc + r + 1, xc - r : xc + r + 1]

    return spots


def extract_picasso_spot_metrics(dat):

    spot_metrics = None
    error = None

    try:

        frame_index = dat["frame_index"]
        box_size = dat["box_size"]
        locs = dat["locs"]
        loc_centers = dat["loc_centers"]
        stop_event = dat["stop_event"]

        if not stop_event.is_set():

            spot_metrics = {}

            spot_cx = [center[0] for center in loc_centers]
            spot_cy = [center[1] for center in loc_centers]

            # Load data from shared memory
            shared_mem = dat["shared_mem"]
            image = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

            frame = image[frame_index].copy()

            camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }

            frame = np.expand_dims(frame, axis=0).copy()

            locs.frame = 0
            spot_data = cut_spots(frame, locs.frame, locs.x, locs.y, box_size)

            # metadata
            spot_metrics["dataset"] = dat["dataset"]
            spot_metrics["channel"] = dat["channel"]
            spot_metrics["frame_index"] = dat["frame_index"]
            spot_metrics["spot_index"] = np.arange(len(locs))
            spot_metrics["spot_cx"] = spot_cx
            spot_metrics["spot_cy"] = spot_cy
            spot_metrics["spot_center"] = loc_centers
            spot_metrics["spot_x"] = locs.x
            spot_metrics["spot_y"] = locs.y
            spot_metrics["box_size"] = dat["box_size"]

            spot_photons = np.zeros(len(locs))
            spot_photons_bg = np.zeros(len(locs))
            spot_sx = np.zeros(len(locs))
            spot_sy = np.zeros(len(locs))
            spot_lpx = np.zeros(len(locs))
            spot_lpy = np.zeros(len(locs))
            spot_net_gradient = np.zeros(len(locs))
            spot_likelihood = np.zeros(len(locs))

            for spot_index, (spot, loc) in enumerate(zip(spot_data, locs)):

                start = time.time()

                while time.time() - start < 5:

                    try:

                        thetas, CRLBs, likelihoods, iterations = gaussmle([spot], eps=0.0001,
                            max_it=500, method="sigma")

                        spot_photons[spot_index] = thetas[:, 2][0]
                        spot_photons_bg[spot_index] = thetas[:, 3][0]
                        spot_sx[spot_index] = thetas[:, 5][0]
                        spot_sy[spot_index] = thetas[:, 4][0]
                        spot_lpx[spot_index] = np.sqrt(CRLBs[:, 1])[0]
                        spot_lpy[spot_index] = np.sqrt(CRLBs[:, 0])[0]
                        spot_net_gradient[spot_index] = loc.net_gradient
                        spot_likelihood[spot_index] = likelihoods[0]

                    except:
                        break

                    break

                if spot_photons[spot_index] <= 0:
                    spot_photons_bg[spot_index] = 0

            spot_metrics["spot_photons"] = spot_photons
            spot_metrics["spot_photons_local_bg"] = spot_photons_bg
            spot_metrics["spot_photons_masked_local_bg"] = spot_photons_bg

            spot_metrics["spot_sx"] = spot_sx
            spot_metrics["spot_sy"] = spot_sy
            spot_metrics["spot_lpx"] = spot_lpx
            spot_metrics["spot_lpy"] = spot_lpy
            spot_metrics["spot_net_gradient"] = spot_net_gradient
            spot_metrics["spot_likelihood"] = spot_likelihood

            spot_metrics = pd.DataFrame(spot_metrics)

    except:
        spot_metrics = None
        error = traceback.format_exc()
        print(traceback.format_exc())

    return spot_metrics

def compute_lsp_background(local_background_arr, background_mean_list):

    try:

        num_frames = local_background_arr.shape[0]
        lsp_backgrounds = np.zeros(num_frames)

        for frame_index, frame, in enumerate(local_background_arr):
            local_pixels = frame.compressed()
            global_mean = background_mean_list[frame_index]

            optimal_x = None
            min_difference = float('inf')
            for percentile in range(20, 80):  # Range based on your given 54-61%
                percentile_value = np.percentile(local_pixels, percentile)
                difference = abs(percentile_value - global_mean)
                if difference < min_difference:
                    min_difference = difference
                    optimal_x = percentile

            lsp_backgrounds[frame_index] = np.percentile(local_pixels, optimal_x)

    except:
        lsp_backgrounds = None
        pass

    return lsp_backgrounds


def extract_background_metrics(dat):

    background_data = None

    try:

        frame = dat["frame"]
        channel = dat["channel"]
        dataset = dat["dataset"]
        shared_mem = dat["shared_mem"]
        stop_event = dat["stop_event"]

        if not stop_event.is_set():

            np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

            background_values = np_array[frame].copy()
            masked_background_values = np.ma.array(background_values, mask=dat["global_spot_mask"])

            n_pixels = dat["n_pixels"]

            spot_mean_global_bg = np.mean(background_values)
            spot_median_global_bg = np.median(background_values)
            spot_sum_global_bg = spot_mean_global_bg * n_pixels
            spot_max_global_bg = np.max(background_values)
            spot_std_global_bg = np.std(background_values)

            spot_mean_masked_global_bg = np.ma.mean(masked_background_values)
            spot_median_masked_global_bg = np.ma.median(masked_background_values)
            spot_sum_masked_global_bg = spot_mean_masked_global_bg * n_pixels
            spot_max_masked_global_bg = np.ma.max(masked_background_values)
            spot_std_masked_global_bg = np.ma.std(masked_background_values)

            background_data = {"dataset": dataset,
                               "channel": channel,
                                "frame_index": frame,
                                "spot_mean_global_bg": spot_mean_global_bg,
                                "spot_median_global_bg": spot_median_global_bg,
                                "spot_sum_global_bg": spot_sum_global_bg,
                                "spot_max_global_bg": spot_max_global_bg,
                                "spot_std_global_bg": spot_std_global_bg,
                                "spot_mean_masked_global_bg": spot_mean_masked_global_bg,
                                "spot_median_masked_global_bg": spot_median_masked_global_bg,
                                "spot_sum_masked_global_bg": spot_sum_masked_global_bg,
                                "spot_max_masked_global_bg": spot_max_masked_global_bg,
                                "spot_std_masked_global_bg": spot_std_masked_global_bg,
                               }

    except:
        print(traceback.format_exc())
        pass

    return background_data

def extract_spot_metrics(dat):

    spot_metrics = None
    frame = None

    try:

        # Load data from shared memory
        shared_mem = dat["shared_mem"]
        np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)
        spot_size = dat["spot_size"]
        spot_center = dat["spot_center"]
        stop_event = dat["stop_event"]
        spot_index = dat["spot_index"]
        channel = dat["channel"]
        frame_shape = np_array.shape[1:]

        if not stop_event.is_set():

            spot_metrics = {}

            spot_cx, spot_cy = spot_center

            n_pixels = spot_size**2

            bounds = dat["spot_bound"]

            spot_mask = dat["spot_mask"]
            spot_mask = spot_mask.astype(np.uint8)
            spot_background_mask = dat["spot_background_mask"]

            [x1, x2, y1, y2], spot_mask, spot_background_mask = crop_spot_data(frame_shape,
                bounds, spot_mask,spot_background_mask)

            spot_overlap = dat["background_overlap_mask"][y1:y2, x1:x2]

            if spot_overlap.shape == spot_background_mask.shape:
                spot_background_mask = spot_background_mask & spot_overlap

            spot_mask = np.logical_not(spot_mask).astype(int)
            spot_background_mask = np.logical_not(spot_background_mask).astype(int)

            spot_loc = dat["spot_loc"]
            spot_x = spot_loc.x
            spot_y = spot_loc.y

            # Perform preprocessing steps and overwrite original image
            spot_values = np_array[:, y1:y2, x1:x2].copy()
            spot_background = np_array[:, y1:y2, x1:x2].copy()
            spot_masked_background = np_array[:, y1:y2, x1:x2].copy()

            spot_mask = np.repeat(spot_mask[np.newaxis, :, :], len(spot_values), axis=0)
            spot_background_mask = np.repeat(spot_background_mask[np.newaxis, :, :], len(spot_masked_background), axis=0)

            spot_values = np.ma.array(spot_values, mask=spot_mask)
            spot_masked_background = np.ma.array(spot_masked_background, mask=spot_background_mask)

            # metadata
            spot_metrics["dataset"] = [dat["dataset"]]*len(spot_values)
            spot_metrics["channel"] = [dat["channel"]]*len(spot_values)
            spot_metrics["frame_index"] = np.arange(len(spot_values)).tolist()
            spot_metrics["spot_index"] = [dat["spot_index"]]*len(spot_values)
            spot_metrics["spot_cx"] = [spot_cx]*len(spot_values)
            spot_metrics["spot_cy"] = [spot_cy]*len(spot_values)
            spot_metrics["spot_x"] = [spot_x]*len(spot_values)
            spot_metrics["spot_y"] = [spot_y]*len(spot_values)
            spot_metrics["spot_size"] = [dat["spot_size"]]*len(spot_values)

            # metrics
            spot_mean = np.ma.mean(spot_values,axis=(1,2)).data
            spot_median = np.ma.median(spot_values,axis=(1,2)).data
            spot_sum = np.ma.sum(spot_values,axis=(1,2)).data
            spot_max = np.ma.max(spot_values,axis=(1,2)).data
            spot_std = np.ma.std(spot_values,axis=(1,2)).data

            spot_mean_local_bg = np.mean(spot_background,axis=(1,2))
            spot_median_local_bg = np.median(spot_background,axis=(1,2))
            spot_sum_local_bg = spot_mean_local_bg*n_pixels
            spot_max_local_bg = np.max(spot_background,axis=(1,2))
            spot_std_local_bg = np.std(spot_background,axis=(1,2))

            spot_mean_masked_local_bg = np.ma.mean(spot_masked_background,axis=(1,2)).data
            spot_median_masked_local_bg = np.ma.median(spot_masked_background,axis=(1,2)).data
            spot_sum_masked_local_bg = spot_mean_masked_local_bg*n_pixels
            spot_max_masked_local_bg = np.ma.max(spot_masked_background,axis=(1,2)).data
            spot_std_masked_local_bg = np.ma.std(spot_masked_background,axis=(1,2)).data

            # populate spot metrics dict
            spot_metrics["spot_mean"] = spot_mean
            spot_metrics["spot_median"] = spot_median
            spot_metrics["spot_sum"] = spot_sum
            spot_metrics["spot_max"] = spot_max
            spot_metrics["spot_std"] = spot_std

            spot_metrics["spot_mean_local_bg"] = spot_mean_local_bg
            spot_metrics["spot_median_local_bg"] = spot_median_local_bg
            spot_metrics["spot_sum_local_bg"] = spot_sum_local_bg
            spot_metrics["spot_max_local_bg"] = spot_max_local_bg
            spot_metrics["spot_std_local_bg"] = spot_std_local_bg

            spot_metrics["spot_mean_masked_local_bg"] = spot_mean_masked_local_bg
            spot_metrics["spot_median_masked_local_bg"] = spot_median_masked_local_bg
            spot_metrics["spot_sum_masked_local_bg"] = spot_sum_masked_local_bg
            spot_metrics["spot_max_masked_local_bg"] = spot_max_masked_local_bg
            spot_metrics["spot_std_masked_local_bg"] = spot_std_masked_local_bg

            n_frames = len(spot_values)

            # populate spot metrics dataframe
            reshaped_spot_metrics = []
            for i in range(n_frames):
                new_dict = {key: spot_metrics[key][i] for key in spot_metrics}
                reshaped_spot_metrics.append(new_dict)

            spot_metrics = pd.DataFrame.from_dict(reshaped_spot_metrics)

    except:
        print(traceback.format_exc())
        spot_metrics = None
        pass

    return spot_metrics


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





class _trace_compute_utils:

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

    def _pixseq_compute_traces_finished(self):

        self.compute_traces.setEnabled(True)

        self.populate_plot_combos()
        self.populate_export_combos()
        self.initialize_plot()

        self.update_ui()

    def _get_bbox_localisations(self, n_frames):

        bbox_locs = None

        try:

            localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])

            locs = copy.deepcopy(localisation_dict["localisations"])

            # Define new dtype including the new field
            new_dtype = np.dtype(locs.dtype.descr + [('loc_index', "<u4")])

            # Create a new array with the new dtype
            extended_locs = np.zeros(locs.shape, dtype=new_dtype)

            for field in locs.dtype.names:
                extended_locs[field] = locs[field]

            extended_locs['loc_index'] = 1

            extended_locs = np.array(extended_locs, dtype=new_dtype).view(np.recarray)

            for loc_index, loc in enumerate(extended_locs):
                loc.loc_index = loc_index

            bbox_locs = []
            for frame_index in range(n_frames):
                frame_locs = copy.deepcopy(extended_locs)
                for loc_index, loc in enumerate(frame_locs):
                    loc.frame = frame_index
                    loc.loc_index = loc.loc_index
                bbox_locs.extend(frame_locs)

            bbox_locs = np.array(bbox_locs, dtype=new_dtype).view(np.recarray)

        except:
            print(traceback.format_exc())
            pass

        return bbox_locs

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

    def compute_background_values(self, image_dict, global_spot_mask):

        try:

            shared_mem = image_dict["shared_mem"]

            background = np.ndarray(image_dict["shape"],
                dtype=image_dict["dtype"], buffer=shared_mem.buf)

            n_frames = background.shape[0]

            mask = np.logical_not(global_spot_mask).astype(int)
            mask = np.repeat(mask[np.newaxis, :, :], n_frames, axis=0)

            masked_background = np.ma.array(background, mask=mask)

            background_values = {
                "background_mean": np.mean(background, axis=(1, 2)),
                "masked_background_mean": np.ma.mean(masked_background, axis=(1, 2)),
            }

        except:
            print(traceback.format_exc())
            pass

        return background_values


    def populate_spot_metric_compute_jobs(self):

        compute_jobs = {"spot_metrics": [],
                        "background_metrics": [],
                        "picasso_metrics": [],
                        }

        try:

            self.traces_spot_size = self.gui.traces_spot_size

            spot_size = int(self.traces_spot_size.currentText())
            spot_shape = self.traces_spot_shape.currentText()
            buffer_size = int(self.traces_background_buffer.currentText())
            bg_width = int(self.traces_background_width.currentText())
            compute_global_background = self.compute_global_background.isChecked()
            compute_picasso = self.compute_with_picasso.isChecked()

            localisation_dict = copy.deepcopy(self.localisation_dict["bounding_boxes"])
            locs = localisation_dict["localisations"].copy()
            box_size = localisation_dict["box_size"]

            spot_mask, buffer_mask, spot_background_mask = self.generate_localisation_mask(spot_size,
                spot_shape, buffer_size, bg_width, plot=False)

            spot_bounds = self.generate_spot_bounds(locs, len(spot_mask[0]))
            spot_centers = self.get_localisation_centres(locs, mode="bounding_boxes")

            spot_metrics_jobs = []
            picasso_metrics_jobs = []
            background_metrics_jobs = []

            for image_dict in self.shared_images:

                mask_shape = image_dict["shape"][1:]
                n_frames = image_dict["shape"][0]
                n_locs = len(locs)
                channel = image_dict["channel"]
                dataset = image_dict["dataset"]
                n_pixels = int(self.traces_spot_size.currentText()) ** 2

                background_overlap_mask, global_spot_mask = self.generate_background_overlap_mask(locs,
                    buffer_mask, spot_background_mask, mask_shape)

                for spot_index, (spot_loc, spot_bound, spot_center) in enumerate(zip(locs, spot_bounds, spot_centers)):
                    spot_compute_task = {"compute_task":"spot_metrics",
                                         "spot_index": spot_index,
                                         "spot_size": spot_size,
                                         "spot_mask": spot_mask,
                                         "spot_background_mask": spot_background_mask,
                                         "global_spot_mask": global_spot_mask,
                                         "background_overlap_mask": background_overlap_mask,
                                         "spot_loc": spot_loc,
                                         "spot_bound": spot_bound,
                                         "spot_center": spot_center,
                                         "stop_event": self.stop_event,
                                         }
                    spot_compute_task = {**spot_compute_task, **image_dict}
                    spot_metrics_jobs.append(spot_compute_task)

                if compute_global_background:
                    for frame in range(n_frames):
                        background_task = {"compute_task":"background_metrics",
                                           "frame": frame,
                                           "channel": channel,
                                           "dataset": dataset,
                                           "n_pixels": n_pixels,
                                           "global_spot_mask": global_spot_mask,
                                           "shared_mem": image_dict["shared_mem"],
                                           "shape": image_dict["shape"],
                                           "dtype": image_dict["dtype"],
                                           "global_spot_mask": global_spot_mask,
                                           "stop_event": self.stop_event,
                                           }
                        background_metrics_jobs.append(background_task)

                if compute_picasso:
                    for frame_index in range(n_frames):
                        picasso_task = {"compute_task":"picasso_metrics",
                                        "frame_index": frame_index,
                                        "channel": channel,
                                        "dataset": dataset,
                                        "n_pixels": n_pixels,
                                        "shared_mem": image_dict["shared_mem"],
                                        "shape": image_dict["shape"],
                                        "dtype": image_dict["dtype"],
                                        "stop_event": self.stop_event,
                                        "locs": copy.deepcopy(locs),
                                        "loc_centers": spot_centers,
                                        "box_size": box_size,
                                        }

                        picasso_metrics_jobs.append(picasso_task)

            compute_jobs["spot_metrics"] = spot_metrics_jobs
            compute_jobs["background_metrics"] = background_metrics_jobs
            compute_jobs["picasso_metrics"] = picasso_metrics_jobs

        except:
            print(traceback.format_exc())
            pass

        return compute_jobs





    def extract_spot_metrics_wrapper(self, progress_callback):

        try:

            compute_global_background = self.compute_global_background.isChecked()
            compute_picasso = self.compute_with_picasso.isChecked()

            compute_jobs = self.populate_spot_metric_compute_jobs()

            spot_metrics_jobs = compute_jobs["spot_metrics"]
            background_metrics_jobs = compute_jobs["background_metrics"]
            picasso_metrics_jobs = compute_jobs["picasso_metrics"]

            spot_metrics = []
            background_metrics = []
            picasso_metrics = []

            cpu_count = int(multiprocessing.cpu_count() * 0.9)

            total_jobs = len(spot_metrics_jobs) + len(background_metrics_jobs) + len(picasso_metrics_jobs)

            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                # Combine both job types into a single dictionary

                futures = {executor.submit(extract_spot_metrics, job): job for job in spot_metrics_jobs}
                if compute_global_background:
                    futures.update({executor.submit(extract_background_metrics, job): job for job in background_metrics_jobs})
                if compute_picasso:
                    futures.update({executor.submit(extract_picasso_spot_metrics, job): job for job in picasso_metrics_jobs})

                iter = 0
                for future in concurrent.futures.as_completed(futures):
                    if self.stop_event.is_set():
                        future.cancel()
                    else:
                        job = futures[future]
                        job_type = job["compute_task"]
                        try:
                            result = future.result()  # Process result here
                            # Append result to the appropriate list based on job type
                            if job_type == "spot_metrics":
                                if result is not None:
                                    spot_metrics.append(result)
                            elif job_type == "picasso_metrics":
                                if result is not None:
                                    picasso_metrics.append(result)
                            else:
                                if result is not None:
                                    background_metrics.append(result)
                        except concurrent.futures.TimeoutError:
                            # Handle timeout
                            pass
                        except Exception as e:
                            print(e)
                            # Handle other exceptions
                            pass

                        # Update progress
                        iter += 1
                        progress = int((iter / total_jobs) * 100)
                        progress_callback.emit(progress)  # Emit the signal

        except:
            self.restore_shared_images()
            self.update_ui()
            print(traceback.format_exc())

        self.spot_metrics = spot_metrics
        self.background_metrics = background_metrics
        self.picasso_spot_metrics = picasso_metrics

        return spot_metrics, background_metrics, picasso_metrics

    def populatate_traces_dict(self):

        spot_metrics = self.spot_metrics
        background_metrics = self.background_metrics
        picasso_spot_metrics = self.picasso_spot_metrics

        try:

            self.traces_dict = {}

            # format spot metrics into dataframe
            if spot_metrics is not None:

                spot_metrics = pd.concat(spot_metrics)
                spot_metrics.sort_values(by=["dataset", "channel", "spot_index", "frame_index"], inplace=True)


            if background_metrics is not None and len(background_metrics) > 0:

                background_metrics = pd.DataFrame(background_metrics)

                merge_keys = ["dataset", "channel", "frame_index"]
                spot_metrics = pd.merge(spot_metrics, background_metrics, on=merge_keys, how='left')


            # format picasso spot metrics into dataframe and merge with spot metrics
            if picasso_spot_metrics is not None and len(picasso_spot_metrics) > 0:

                picasso_spot_metrics = pd.concat(picasso_spot_metrics)
                picasso_spot_metrics.sort_values(by=["dataset", "channel", "spot_index", "frame_index"], inplace=True)

                merge_keys = ["dataset", "channel", "spot_index","frame_index", "spot_cx","spot_cy"]
                spot_metrics = pd.merge(spot_metrics, picasso_spot_metrics, on=merge_keys, how='left')

            # populate traces dict
            if spot_metrics is not None:

                for names, data in spot_metrics.groupby(["dataset", "channel", "spot_index"]):

                    dataset, channel, spot_index = names

                    channel_dict = self.dataset_dict[dataset][channel]

                    if "gap_label" in channel_dict.keys():
                        gap_label = self.dataset_dict[dataset][channel]["gap_label"]
                        sequence_label = self.dataset_dict[dataset][channel]["sequence_label"]
                    else:
                        gap_label = None
                        sequence_label = None

                    if dataset not in self.traces_dict.keys():
                        self.traces_dict[dataset] = {}
                    if channel not in self.traces_dict[dataset].keys():
                        self.traces_dict[dataset][channel] = {}
                    if spot_index not in self.traces_dict[dataset][channel].keys():
                        self.traces_dict[dataset][channel][spot_index] = {}

                    for column in data.columns:
                        if column not in ["dataset", "channel", "spot_index", "frame_index"]:
                            self.traces_dict[dataset][channel][spot_index][column] = data[column].values

                        # add gap label and sequence label
                        self.traces_dict[dataset][channel][spot_index]["gap_label"] = gap_label
                        self.traces_dict[dataset][channel][spot_index]["sequence_label"] = sequence_label

        except:
            print(traceback.format_exc())
            pass

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

        except:
            print(traceback.format_exc())
            pass


    def _pixseq_compute_traces(self, progress_callback=None, picasso=False):

        try:

            self.spot_metrics = None
            self.background_metrics = None
            self.picasso_spot_metrics = None

            self.shared_images = self.create_shared_images()

            self.extract_spot_metrics_wrapper(progress_callback)

            self.restore_shared_images()

            self.populatate_traces_dict()

            self.compute_photo_bleaching()
            self.compute_traces.setEnabled(True)

        except:
            self.update_ui()
            self.restore_shared_images()
            print(traceback.format_exc())
            pass

    def pixseq_compute_traces(self):

        try:

            compute_traces = False

            if self.localisation_dict != {}:
                if "bounding_boxes" in self.localisation_dict.keys():
                    if "localisations" in self.localisation_dict["bounding_boxes"].keys():
                        n_bboxes = len(self.localisation_dict["bounding_boxes"]["localisations"])

                        if n_bboxes > 0:
                            compute_traces = True

            if compute_traces == True:

                self.pixseq_notification(f"Computing traces for {n_bboxes} bounding boxes.")

                self.update_ui(init=True)

                self.worker = Worker(self._pixseq_compute_traces)
                self.worker.signals.progress.connect(partial(self.pixseq_progress, progress_bar=self.compute_traces_progressbar))
                self.worker.signals.finished.connect(self._pixseq_compute_traces_finished)
                self.worker.signals.error.connect(self._pixseq_compute_traces_finished)
                self.threadpool.start(self.worker)

            else:
                self.pixseq_notification("Bounding Boxes required for trace computation.")


        except:
            self.update_ui()
            self.restore_shared_images()
            print(traceback.format_exc())

    def visualise_background_masks(self):

        self.compute_traces.setEnabled(True)

        try:

            import cv2

            if "bounding_boxes" in self.localisation_dict.keys():
                if "localisations" in self.localisation_dict["bounding_boxes"].keys():

                    spot_size = int(self.traces_spot_size.currentText())
                    spot_shape = self.traces_spot_shape.currentText()
                    buffer_size = int(self.traces_background_buffer.currentText())
                    bg_width = int(self.traces_background_width.currentText())

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
            import cv2

            if "bounding_boxes" in self.localisation_dict.keys():
                if "fitted" in self.localisation_dict["bounding_boxes"].keys():

                    spot_size = int(self.traces_spot_size.currentText())
                    spot_shape = self.traces_spot_shape.currentText()
                    buffer_size = int(self.traces_background_buffer.currentText())
                    bg_width = int(self.traces_background_width.currentText())

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




camera_info = {"baseline": 100.0, "gain": 1, "sensitivity": 1.0, "qe": 0.9, }

# method = "lq"
# gain = 1
#
# total_locs = 0
# progress_dict = {}
# for image_index, image_dict in enumerate(self.shared_images):
#     total_locs += image_dict["n_frames"] * len(locs)
#     if image_index not in progress_dict:
#         progress_dict[image_index] = 0
#
#
# for image_index, image_dict in enumerate(self.shared_images):
#
#     n_frames = image_dict["n_frames"]
#
#     image_data = np.ndarray(image_dict["shape"],
#         dtype=image_dict["dtype"], buffer=image_dict["shared_mem"].buf)
#
#     image_locs = []
#
#     for frame_index in range(n_frames):
#         locs_copy = copy.deepcopy(locs)
#         for loc in locs_copy:
#             loc.frame = frame_index
#             image_locs.append(loc)
#
#     image_locs = np.rec.fromrecords(image_locs, dtype=locs.dtype)
#
#     detected_loc_spots = localize.get_spots(image_data,
#         image_locs, box_size, camera_info)
#
#     fs = gausslq.fit_spots_parallel(detected_loc_spots, asynch=True)
#
#     n_tasks = len(fs)
#     while lib.n_futures_done(fs) < n_tasks:
#         progress = (lib.n_futures_done(fs) / n_tasks) * 100
#         progress_dict[image_index] = progress
#         total_progress = int(np.sum(list(progress_dict.values())) / total_locs)
#         progress_callback.emit(total_progress)
#         time.sleep(0.1)
#
#     theta = gausslq.fits_from_futures(fs)
#     em = gain > 1
#
#     fitted_locs = gausslq.locs_from_fits(image_locs, theta, box_size, em)