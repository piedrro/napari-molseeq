from napari_pixseq.funcs.pixseq_utils_compute import Worker
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import concurrent
import os
import h5py
import yaml
import tempfile
import shutil
from pathlib import Path
import traceback
import numpy as np
from qtpy.QtWidgets import QFileDialog

class picasso_loc_utils():

    def __init__(self, locs: np.recarray = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.locs = locs

        self.detected_dtype = [('frame', '<i4'),
                               ('x', '<i4'),
                               ('y', '<i4'),
                               ('net_gradient', '<f4')]

        self.fitted_dtype = [('frame', '<u4'),
                           ('x', '<f4'),
                           ('y', '<f4'),
                           ('photons', '<f4'),
                           ('sx', '<f4'),
                           ('sy', '<f4'),
                           ('bg', '<f4'),
                           ('lpx', '<f4'),
                           ('lpy', '<f4'),
                           ('ellipticity', '<f4'),
                           ('net_gradient', '<f4')]

        if self.locs is not None:
            self.get_loc_info()

    def get_loc_info(self):

        self.dtype = self.locs.dtype
        self.columns = self.locs.dtype.names

        if self.locs.dtype == self.fitted_dtype:
            self.loc_type = "fiducial"
        else:
            self.loc_type = "bbox"

    def coerce_new_loc_format(self, new_loc):

        if len(new_loc) != len(self.dtype):
            difference = len(self.dtype) - len(new_loc)
            if difference > 0:
                new_loc = list(new_loc)
                for i in range(difference):
                    new_loc = new_loc + [0]
                new_loc = tuple(new_loc)

        return new_loc

    def remove_duplicate_locs(self, locs = None):

            try:

                if locs is not None:
                    self.locs = locs
                    self.get_loc_info()

                unique_records, indices = np.unique(self.locs, return_index=True)

                self.locs = self.locs[indices]

            except:
                print(traceback.format_exc())
                pass

            return locs


    def add_loc(self, locs = None, new_loc = None):

        try:

            if locs is not None:
                self.locs = locs
                self.get_loc_info()

            if type(new_loc) == list:
                new_loc = tuple(new_loc)
            if type(new_loc) in [np.ndarray, np.recarray]:
                new_loc = tuple(new_loc.tolist())

            new_loc = self.coerce_new_loc_format(new_loc)

            self.locs = np.array(self.locs).tolist()
            self.locs.append(new_loc)
            self.locs = np.rec.fromrecords(self.locs, dtype=self.dtype)

            self.remove_duplicate_locs()

        except:
            print(traceback.format_exc())
            pass

        return self.locs

    def remove_loc(self, locs = None, loc_index = None):

        try:

            if locs is not None:
                self.locs = locs
                self.get_loc_info()

            if loc_index is not None:

                if loc_index < len(self.locs):

                    self.locs = self.locs.view(np.float32).reshape(len(self.locs), -1)
                    self.locs = np.delete(self.locs, loc_index, axis=0)
                    self.locs = self.locs.view(self.dtype)
                    self.locs = np.squeeze(self.locs, axis=1)

                self.remove_duplicate_locs()

        except:
            print(loc_index)
            print(self.locs, self.locs.dtype)
            print(traceback.format_exc())
            pass

        return self.locs

    def create_locs(self, new_loc, fitted = False):

        try:

            if fitted == False:
                dtype = self.detected_dtype
            else:
                dtype = self.fitted_dtype

            if type(new_loc) == list:
                new_loc = [tuple(new_loc)]
            elif type(new_loc) == tuple:
                new_loc = [new_loc]
            elif type(new_loc) in [np.array, np.ndarray]:
                new_loc = [tuple(new_loc.tolist())]

            self.locs = np.rec.fromrecords(new_loc, dtype=dtype)

        except:
            print(traceback.format_exc())
            pass

        return self.locs


def format_picasso_path(path):

    if "%" in str(path):
        path = path.replace("%", "%%")

    path = os.path.normpath(path)

    if os.name == "nt":
        if path.startswith("\\\\"):
            path = '\\\\?\\UNC\\' + path[2:]

    return Path(path)

def export_picasso_localisation(loc_data):

    try:
        locs = loc_data["locs"]
        h5py_path = loc_data["hdf5_path"]
        yaml_path = loc_data["info_path"]
        info = loc_data["picasso_info"]

        h5py_path = format_picasso_path(h5py_path)
        yaml_path = format_picasso_path(yaml_path)

        # Create temporary files
        temp_h5py_path = tempfile.NamedTemporaryFile(delete=False).name
        temp_yaml_path = tempfile.NamedTemporaryFile(delete=False).name

        h5py_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to temporary HDF5 file
        with h5py.File(temp_h5py_path, "w") as hdf_file:
            hdf_file.create_dataset("locs", data=locs)

        # Save to temporary YAML file
        with open(temp_yaml_path, "w") as file:
            yaml.dump_all(info, file, default_flow_style=False)

        try:
            shutil.move(temp_h5py_path, h5py_path)
            shutil.move(temp_yaml_path, yaml_path)
        except:

            print("Could not move files to import directory. Saving to desktop instead.")

            desktop_dir = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

            desktop_h5py_path = os.path.join(desktop_dir, h5py_path.name)
            desktop_yaml_path = os.path.join(desktop_dir, yaml_path.name)

            shutil.move(temp_h5py_path, desktop_h5py_path)
            shutil.move(temp_yaml_path, desktop_yaml_path)

    except Exception as e:
        print(traceback.format_exc())

class _loc_utils():


    def _import_picasso_localisations_finished(self):

        try:
            self.update_ui()

            self.draw_fiducials(update_vis=True)
            self.draw_bounding_boxes(update_vis=True)

        except:
            print(traceback.format_exc())
            pass


    def _import_picasso_localisations(self, progress_callback = None, path=""):

        try:

            dataset = self.import_picasso_dataset.currentText()
            channel = self.import_picasso_channel.currentText()
            type = self.import_picasso_type.currentText()

            if type == "Fiducials":
                if dataset not in self.localisation_dict["fiducials"].keys():
                    self.localisation_dict["fiducials"][dataset] = {}
                if channel.lower() not in self.localisation_dict["fiducials"][dataset].keys():
                    self.localisation_dict["fiducials"][dataset][channel.lower()] = {}

            yaml_path = path.replace(".hdf5", ".yaml")

            dtype = [('frame', '<u4'), ('x', '<f4'), ('y', '<f4'),
                     ('photons', '<f4'), ('sx', '<f4'), ('sy', '<f4'),
                     ('bg', '<f4'), ('lpx', '<f4'), ('lpy', '<f4'),
                     ('ellipticity', '<f4'),('net_gradient', '<f4')]

            if self.verbose:
                print("Loading localisations from hdf5")

            with h5py.File(path, "r") as f:
                locs = np.array(f["locs"], dtype=dtype).view(np.recarray)
                # print(locs.dtype.descr)
                # print(len(locs[0]))

            box_size = self.picasso_box_size.currentText()


            if self.verbose:
                print("Loading localisations from yaml")

            if os.path.exists(yaml_path):
                with open(yaml_path, "r") as info_file:
                    info = list(yaml.load_all(info_file, Loader=yaml.UnsafeLoader))

                if "Box Size" in info[1].keys():
                    box_size = info[1]["Box Size"]

            if type == "Fiducials":

                if self.verbose:
                    print("Creating render locs")

                render_locs = {}
                for frame in np.unique(locs.frame):
                    frame_locs = locs[locs.frame == frame].copy()
                    render_locs[frame] = np.vstack((frame_locs.y, frame_locs.x)).T.tolist()

                loc_centres = self.get_localisation_centres(locs)

                if self.verbose:
                    print("Updating localisation dict")

                self.localisation_dict["fiducials"][dataset][channel.lower()]["localisations"] = locs.copy()
                self.localisation_dict["fiducials"][dataset][channel.lower()]["localisation_centres"] = loc_centres.copy()
                self.localisation_dict["fiducials"][dataset][channel.lower()]["render_locs"] = render_locs
                self.localisation_dict["fiducials"][dataset][channel.lower()]["fitted"] = True
                self.localisation_dict["fiducials"][dataset][channel.lower()]["box_size"] = box_size

            else:
                unique_frames = np.unique(locs.frame)
                locs = locs[locs.frame == unique_frames[0]]

                loc_centres = self.get_localisation_centres(locs)

                if self.verbose:
                    print("Updating localisation dict")

                self.localisation_dict["bounding_boxes"]["localisations"] = locs.copy()
                self.localisation_dict["bounding_boxes"]["localisation_centres"] = loc_centres.copy()
                self.localisation_dict["bounding_boxes"]["render_locs"] = {}
                self.localisation_dict["bounding_boxes"]["fitted"] = True
                self.localisation_dict["bounding_boxes"]["box_size"] = box_size

        except:
            print(traceback.format_exc())
            pass


    def import_picaaso_localisations(self):

        try:

            dataset = self.import_picasso_dataset.currentText()
            channel = self.import_picasso_channel.currentText()
            type = self.import_picasso_type.currentText()

            if dataset in self.dataset_dict.keys():
                if channel.lower() in self.dataset_dict[dataset].keys():

                    dataset_path = self.dataset_dict[dataset][channel.lower()]["path"]

                    dataset_dir = os.path.dirname(dataset_path)

                    if os.path.exists(dataset_dir) == False:
                        dataset_dir = os.path.expanduser("~/Desktop")

                    path, _ = QFileDialog.getOpenFileName(self, "Open Files", dataset_dir, "Files (*.hdf5)")

                    if path != "":

                        path = self.format_import_path(path)

                        if os.path.exists(path):

                            self.update_ui(init=True)

                            self.worker = Worker(self._import_picasso_localisations, path=path)
                            self.worker.signals.finished.connect(self._import_picasso_localisations_finished)
                            self.threadpool.start(self.worker)
        except:
            self.update_ui()
            print(traceback.format_exc())
            pass

    def update_loc_export_options(self):

        try:

            dataset_name  = self.locs_export_dataset.currentText()

            if dataset_name in self.dataset_dict.keys() or dataset_name == "All Datasets":

                if dataset_name == "All Datasets":
                    channel_names = ["All Channels"]
                else:
                    channel_names = list(self.dataset_dict[dataset_name].keys())
                    channel_names = [name for name in channel_names if "efficiency" not in name.lower()]

                    for channel_index, channel_name in enumerate(channel_names):
                        if channel_name in ["donor", "acceptor"]:
                            channel_names[channel_index] = channel_name.capitalize()
                        else:
                            channel_names[channel_index] = channel_name.upper()

                    channel_names.insert(0, "All Channels")

                self.locs_export_channel.blockSignals(True)
                self.locs_export_channel.clear()
                self.locs_export_channel.addItems(channel_names)
                self.locs_export_channel.blockSignals(False)

        except:
            print(traceback.format_exc())
            pass


    def export_locs(self, progress_callback = None, export_dataset = "", export_channel = ""):

        try:

            export_loc_mode = self.locs_export_mode.currentText()
            export_loc_jobs = []

            if export_dataset == "All Datasets":
                dataset_list = list(self.dataset_dict.keys())
            else:
                dataset_list = [export_dataset]

            if export_loc_mode == "Fiducials":
                loc_type_list = ["Fiducials"]
            elif export_loc_mode == "Bounding Boxes":
                loc_type_list = ["Bounding Boxes"]
            else:
                loc_type_list = ["Fiducials", "Bounding Boxes"]

            for dataset_name in dataset_list:

                if export_channel == "All Channels":
                    channel_list = list(self.dataset_dict[dataset_name].keys())
                else:
                    channel_list = [export_channel]

                channel_list = [channel.lower() for channel in channel_list if "efficiency" not in channel.lower()]

                for channel_name in channel_list:

                    for loc_type in loc_type_list:

                        if loc_type == "Fiducials":
                            loc_dict, n_locs, fitted = self.get_loc_dict(dataset_name, channel_name, type="fiducials")
                        elif loc_type == "Bounding Boxes":
                            loc_dict, n_locs, fitted = self.get_loc_dict(dataset_name, channel_name, type="bounding_boxes")

                        if n_locs > 0 and fitted == True:

                            locs = loc_dict["localisations"]
                            box_size = loc_dict["box_size"]

                            if "min_net_gradient" in loc_dict.keys():
                                min_net_gradient = loc_dict["min_net_gradient"]
                            else:
                                min_net_gradient = int(self.picasso_min_net_gradient.text())

                            if channel_name in self.dataset_dict[dataset_name].keys():

                                import_path = self.dataset_dict[dataset_name][channel_name]["path"]
                                image_shape = self.dataset_dict[dataset_name][channel_name]["data"].shape

                                base, ext = os.path.splitext(import_path)

                                if channel_name in ["donor", "acceptor"]:
                                    export_channel_name = channel_name.capitalize()
                                else:
                                    export_channel_name = channel_name.upper()

                                if loc_type == "Bounding Boxes":
                                    hdf5_path = base + f"_picasso_bboxes.hdf5"
                                    info_path = base + f"_picasso_bboxes.yaml"
                                else:
                                    hdf5_path = base + f"_picasso_{export_channel_name}_fiducials.hdf5"
                                    info_path = base + f"_picasso_{export_channel_name}_fiducials.yaml"

                                picasso_info = [{"Byte Order": "<", "Data Type": "uint16", "File": import_path,
                                                 "Frames": image_shape[0], "Height": image_shape[1],
                                                 "Micro-Manager Acquisiton Comments": "", "Width":image_shape[2],},
                                                {"Box Size": box_size, "Fit method": "LQ, Gaussian", "Generated by": "Picasso Localize",
                                                 "Min. Net Gradient": min_net_gradient, "Pixelsize": 130, "ROI": None, }]

                                export_loc_job = { "dataset_name": dataset_name,
                                                   "channel_name": channel_name,
                                                   "loc_type": loc_type,
                                                   "locs": locs,
                                                   "fitted": fitted,
                                                   "hdf5_path": hdf5_path,
                                                   "info_path": info_path,
                                                   "picasso_info": picasso_info,
                                                  }
                            export_loc_jobs.append(export_loc_job)

            if len(export_loc_jobs) > 0:

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    futures = [executor.submit(export_picasso_localisation, job) for job in export_loc_jobs]

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result()
                        except:
                            print(traceback.format_exc())
                            pass

                        progress = int(100 * (len(export_loc_jobs) - len(futures)) / len(export_loc_jobs))

                        if progress_callback is not None:
                            progress_callback.emit(progress)




        except:
            self.update_ui()
            print(traceback.format_exc())
            pass


    def export_locs_finished(self):

        try:

            print("Exporting locs finished")
            self.update_ui()

        except:
            self.update_ui()
            print(traceback.format_exc())
            pass

    def initialise_export_locs(self, event=None, export_dataset = "", export_channel = ""):

        try:

            if export_dataset == "" or export_dataset not in self.dataset_dict.keys():
                export_dataset = self.locs_export_dataset.currentText()
            if export_channel == "":
                export_channel = self.locs_export_channel.currentText()

            self.update_ui(init = True)

            self.worker = Worker(self.export_locs, export_dataset = export_dataset, export_channel = export_channel)
            self.worker.signals.progress.connect(partial(self.pixseq_progress,progress_bar=self.export_progressbar))
            self.worker.signals.finished.connect(self.export_locs_finished)
            self.threadpool.start(self.worker)

        except:
            self.update_ui()
            pass

    def get_loc_dict(self, dataset_name="", channel_name="", type = "fiducials"):

        loc_dict = {}
        n_localisations = 0
        fitted = False

        try:

            if type.lower() == "fiducials":

                if dataset_name not in self.localisation_dict["fiducials"].keys():
                    self.localisation_dict["fiducials"][dataset_name] = {}
                else:
                    if channel_name not in self.localisation_dict["fiducials"][dataset_name].keys():
                        self.localisation_dict["fiducials"][dataset_name][channel_name] = {}
                    else:
                        loc_dict = self.localisation_dict["fiducials"][dataset_name][channel_name].copy()

            else:

                if "bounding_boxes" not in self.localisation_dict.keys():
                    self.localisation_dict["bounding_boxes"] = {}

                loc_dict = self.localisation_dict["bounding_boxes"].copy()

            if "localisations" in loc_dict.keys():
                n_localisations = len(loc_dict["localisations"])

            if "fitted" in loc_dict.keys():
                fitted = loc_dict["fitted"]

        except:
            print(traceback.format_exc())
            pass

        return loc_dict, n_localisations, fitted


    def update_loc_dict(self, dataset_name="", channel_name="", type = "fiducials", loc_dict = {}):

        try:

            if type == "fiducials":
                self.localisation_dict["fiducials"][dataset_name][channel_name] = loc_dict
            else:
                self.localisation_dict["bounding_boxes"] = loc_dict

        except:
            print(traceback.format_exc())
            pass

    def get_bbox_dict(self,dataset_name, channel_name):

        bbox_dict = {}

        if "bounding_boxes" not in self.localisation_dict.keys():
            self.localisation_dict["bounding_boxes"] = {}

        return bbox_dict

    def compute_net_gradient(self, position, box_size = None):

        net_gradient = 0

        try:

            dataset = self.pixseq_dataset_selector.currentText()
            channel = self.active_channel
            frame = self.viewer.dims.current_step[0]

            if box_size is None:
                box_size = self.picasso_box_size.currentText()

            loc_mask, _, loc_bg_mask = self.generate_localisation_mask(
                box_size, spot_shape = "square")

            box_size = len(loc_mask[0])

            x, y = position[0], position[1]

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
                x2 = x + (box_size // 2) + 1
                y1 = y - (box_size // 2)
                y2 = y + (box_size // 2) + 1

            loc_spot = self.dataset_dict[dataset][channel]["data"][frame][y1:y2, x1:x2]

            loc_spot_values = np.sum(loc_spot[loc_mask])
            loc_spot_bg_values = np.mean(loc_spot[loc_bg_mask])

            net_gradient = loc_spot_values

        except:
            print(traceback.format_exc())
            pass

        return float(net_gradient)



    def add_manual_localisation(self, position, mode):

        try:

            active_dataset = self.pixseq_dataset_selector.currentText()
            active_channel = self.active_channel
            box_size = int(self.picasso_box_size.currentText())
            frame = self.viewer.dims.current_step[0]
            net_gradient = self.compute_net_gradient(position, box_size=box_size)

            if mode == "fiducials":

                loc_dict, n_locs, _ = self.get_loc_dict(active_dataset, active_channel,
                    type = "fiducials")


                if n_locs > 0:
                    locs = loc_dict["localisations"].copy()
                    render_locs = loc_dict["render_locs"].copy()
                    loc_centers = loc_dict["localisation_centres"].copy()
                    box_size = int(loc_dict["box_size"])
                    dtype = locs.dtype

                    loc_utils = picasso_loc_utils(locs)

                    x,y = position

                    frame_locs = locs[locs.frame == frame]

                    if len(frame_locs) > 0:

                        loc_coords = np.vstack((frame_locs.y, frame_locs.x)).T

                        # Calculate Euclidean distances
                        distances = np.sqrt(np.sum((loc_coords - np.array([y,x])) ** 2, axis=1))

                        # Find the index of the minimum distance
                        min_index = np.argmin(distances)
                        min_distance = distances[min_index]

                        if min_distance < box_size:

                            locs = loc_utils.remove_loc(loc_index=min_index)

                            loc_centers = np.delete(loc_centers, min_index, axis=0)
                            loc_centers = loc_centers.tolist()

                            render_frame_locs = render_locs[frame].copy()
                            render_frame_locs = np.unique(render_frame_locs, axis=0).tolist()
                            distances = np.sqrt(np.sum((np.array(render_frame_locs) - np.array([y,x])) ** 2, axis=1))
                            min_index = np.argmin(distances)
                            render_frame_locs.pop(min_index)
                            render_locs[frame] = render_frame_locs

                            loc_dict["localisations"] = locs
                            loc_dict["localisation_centres"] = loc_centers
                            loc_dict["render_locs"] = render_locs

                            self.update_loc_dict(active_dataset, active_channel, "fiducials", loc_dict)
                            self.draw_fiducials(update_vis=True)

                        else:

                            locs = loc_utils.add_loc(new_loc = [frame, x, y, net_gradient])

                            loc_centers = np.append(loc_centers, np.array([[frame,y,x]], dtype=int), axis=0)
                            loc_centers = loc_centers.tolist()

                            if len(render_locs[frame]) == 0:
                                render_locs[frame] = [[round(y),round(x)]]
                            else:
                                render_locs[frame].append([round(y),round(x)])

                            loc_dict["localisations"] = locs
                            loc_dict["localisation_centres"] = loc_centers
                            loc_dict["render_locs"] = render_locs

                            self.update_loc_dict(active_dataset, active_channel, "fiducials", loc_dict)
                            self.draw_fiducials(update_vis=True)

                    else:

                        locs = loc_utils.add_loc(new_loc=[frame, x, y, net_gradient])

                        loc_centers = np.append(loc_centers, np.array([[frame, y, x]], dtype=int), axis=0)
                        loc_centers = loc_centers.tolist()

                        if frame in render_locs.keys():
                            if len(render_locs[frame]) == 0:
                                render_locs[frame] = [[round(y), round(x)]]
                            else:
                                render_locs[frame].append([round(y), round(x)])
                        else:
                            render_locs[frame] = [[round(y), round(x)]]

                        loc_dict["localisations"] = locs
                        loc_dict["localisation_centres"] = loc_centers
                        loc_dict["render_locs"] = render_locs

                        self.update_loc_dict(active_dataset, active_channel, "fiducials", loc_dict)
                        self.draw_fiducials(update_vis=True)

                else:
                    x, y = position

                    box_size = int(self.picasso_box_size.currentText())

                    new_loc = [frame, position[0], position[1], net_gradient]

                    loc_utils = picasso_loc_utils()
                    locs = loc_utils.create_locs(new_loc=new_loc)

                    loc_centers = [[frame, y, x]]
                    render_locs = {frame: [[y, x]]}

                    loc_dict["localisations"] = locs
                    loc_dict["localisation_centres"] = loc_centers
                    loc_dict["render_locs"] = render_locs
                    loc_dict["fitted"] = False
                    loc_dict["box_size"] = box_size

                    self.update_loc_dict(active_dataset, active_channel, "fiducials", loc_dict)
                    self.draw_fiducials(update_vis=True)

            elif mode == "bounding_box":

                loc_dict, n_locs,_ = self.get_loc_dict(active_dataset, active_channel,
                    type = "bounding_box")

                if n_locs > 0:

                    locs = loc_dict["localisations"].copy()
                    loc_centers = loc_dict["localisation_centres"].copy()
                    box_size = int(loc_dict["box_size"])
                    dtype = locs.dtype

                    loc_utils = picasso_loc_utils(locs)

                    x, y = position

                    loc_centers = np.array(loc_centers).copy()

                    if loc_centers.shape[-1] != 2:
                        loc_coords = loc_centers[:, 1:].copy()
                    else:
                        loc_coords = loc_centers.copy()

                    # Calculate Euclidean distances
                    distances = np.sqrt(np.sum((loc_coords - np.array([y, x])) ** 2, axis=1))

                    # Find the index of the minimum distance
                    min_index = np.argmin(distances)
                    min_distance = distances[min_index]

                    # print(f"min_distance: {min_distance}")

                    if min_distance < box_size:

                        locs = loc_utils.remove_loc(loc_index=min_index)

                        loc_centers = np.delete(loc_centers, min_index, axis=0)
                        loc_centers = loc_centers.tolist()

                        # print(f"len locs: {len(locs)}, len loc_centers: {len(loc_centers)}")

                        loc_dict["localisations"] = locs
                        loc_dict["localisation_centres"] = loc_centers
                        self.update_loc_dict(active_dataset, active_channel, "bounding_boxes", loc_dict)
                        self.draw_bounding_boxes()

                    else:

                        locs = loc_utils.add_loc(new_loc = [frame, x, y, net_gradient])

                        if loc_centers.shape[-1] == 3:
                            loc_centers = np.append(loc_centers, np.array([[frame, y,x]], dtype=int), axis=0)
                        if loc_centers.shape[-1] == 2:
                            loc_centers = np.append(loc_centers, np.array([[y,x]], dtype=int), axis=0)

                        loc_centers = loc_centers.tolist()

                        loc_dict["localisations"] = locs
                        loc_dict["localisation_centres"] = loc_centers

                        self.update_loc_dict(active_dataset, active_channel, "bounding_boxes", loc_dict)
                        self.draw_bounding_boxes()

                else:

                    x, y = position

                    box_size = int(self.picasso_box_size.currentText())

                    new_loc = [frame, position[0], position[1], net_gradient]

                    loc_utils = picasso_loc_utils()
                    locs = loc_utils.create_locs(new_loc=new_loc)

                    loc_centers = [[y, x]]

                    loc_dict["localisations"] = locs
                    loc_dict["localisation_centres"] = loc_centers
                    loc_dict["fitted"] = False
                    loc_dict["box_size"] = box_size

                    self.update_loc_dict(active_dataset, active_channel, "bounding_boxes", loc_dict)
                    self.draw_bounding_boxes()

            elif mode == "lsp":

                x, y = position

                print(x,y)

        except:
            print(traceback.format_exc())
            pass


