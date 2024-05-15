import traceback
import numpy as np
import os
from PIL import Image
from qtpy.QtWidgets import QFileDialog
from napari_pixseq.funcs.pixseq_utils_compute import Worker
import time
import multiprocessing
from multiprocessing import shared_memory, Manager
from functools import partial
import tifffile
import concurrent.futures


def import_image_data(dat, progress_dict={}, index=0):

    try:

        path = dat["path"]
        frame_list = dat["frame_list"]
        channel_list = dat["channel_list"]
        channel_frame_list = dat["channel_frame_list"]
        channel_images = dat["channel_images"]

        n_frames = len(frame_list)

        with Image.open(path) as image:

            for (frame_index, channels, channel_frame) in zip(frame_list, channel_list, channel_frame_list):

                image_shape = dat["image_shape"]
                stop_event = dat["stop_event"]

                if not stop_event.is_set():

                    image.seek(frame_index)
                    img_frame = np.array(image)

                    if len(channels) == 1:
                        img_frames = [img_frame]
                    else:
                        img_frames = np.array_split(img_frame, 2, axis=-1)

                    for channel, channel_img in zip(channels, img_frames):

                        shared_mem = channel_images[channel]
                        np_array = np.ndarray(image_shape, dtype=dat["dtype"], buffer=shared_mem.buf)
                        np_array[channel_frame] = channel_img

                progress = int(((frame_index + 1) / n_frames)*100)
                progress_dict[index] = progress

    except:
        print(traceback.format_exc())
        pass



class _import_utils:

    def create_import_shared_image(self, image_size):

        shared_mem = None

        try:

            if self.verbose:
                print("Creating shared image...")

            shared_mem = shared_memory.SharedMemory(create=True, size=image_size)

        except:
            print(traceback.format_exc())
            pass

        return shared_mem

    def get_image_info(self, path):

        if self.verbose:
            print(f"Getting image info for {path}")

        image_size = os.path.getsize(path)  # Get file size directly

        with tifffile.TiffFile(path) as tif:
            n_frames = len(tif.pages)  # Number of pages (frames)
            page_shape = tif.pages[0].shape  # Dimensions of the first page
            dtype = tif.pages[0].dtype  # Data type of the first page

        image_shape = (n_frames, page_shape[0], page_shape[1])

        return n_frames, image_shape, dtype, image_size

    def format_import_path(self, path):

        try:

            path = os.path.normpath(path)

            if os.name == "nt":
                if path.startswith("\\\\"):
                    path = '\\\\?\\UNC\\' + path[2:]

                    if "%" in str(path):
                        path = path.replace("%", "%%")

                if path.startswith("UNC"):
                    path = '\\\\?\\' + path

                    if "%" in str(path):
                        path = path.replace("%", "%%")

        except:
            print(traceback.format_exc())
            pass

        return path

    def populate_import_lists(self, progress_callback=None, paths=[]):

        image_list = []
        import_dict = {}
        shared_images = {}

        try:

            if self.verbose:
                print("Populating import lists/metadata...")

            import_mode = self.gui.pixseq_import_mode.currentText()
            import_limit_combo = self.gui.pixseq_import_limt.currentText()
            channel_layout = self.gui.pixseq_channel_layout.currentText()
            alex_first_frame = self.gui.pixseq_alex_first_frame.currentText()

            for path_index, path in enumerate(paths):

                path = self.format_import_path(path)
                file_name = os.path.basename(path)

                if self.gui.pixseq_append.isChecked():
                    dataset_name = self.gui.pixseq_append_dataset.currentText()
                else:
                    dataset_name = file_name

                if dataset_name not in shared_images.keys():
                    shared_images[dataset_name] = {}

                n_frames, image_shape, dtype, image_size = self.get_image_info(path)

                if import_mode.lower() in ["donor", "acceptor", "dd", "da", "ad", "aa"]:

                    if import_limit_combo != "None":
                        import_limit = int(self.gui.pixseq_import_limt.currentText())
                    else:
                        import_limit = n_frames

                    image_shape = (import_limit, image_shape[1], image_shape[2])

                    frame_list = list(range(n_frames))[:import_limit]

                    unique_frames = np.unique(frame_list)
                    n_frames = len(unique_frames)

                    channel_names = [import_mode.lower()]
                    channel_list = [channel_names] * n_frames

                    channel_images = {}
                    for channel in channel_names:

                        if self.verbose:
                            print(f"Creating image memory for {dataset_name} {channel}...")

                        shared_image = self.create_import_shared_image(image_size)
                        channel_images[channel] = shared_image
                        shared_images[dataset_name][channel] = shared_image

                    image_dict = {"path": path,
                                  "dataset_name": dataset_name,
                                  "n_frames": n_frames,
                                  "channel_names": channel_names,
                                  "channel_list": channel_list,
                                  "frame_list": frame_list,
                                  "channel_frame_list": frame_list,
                                  "channel_images": channel_images,
                                  "image_shape": image_shape,
                                  "channel_layout": channel_layout,
                                  "alex_first_frame": alex_first_frame,
                                  "dtype": dtype,
                                  "import_mode": import_mode.lower()}

                    image_list.append(image_dict)

                elif import_mode.lower() == "single channel":

                    if import_limit_combo != "None":
                        import_limit = int(self.gui.pixseq_import_limt.currentText())
                    else:
                        import_limit = n_frames

                    image_shape = (import_limit, image_shape[1], image_shape[2])

                    frame_list = list(range(n_frames))[:import_limit]

                    unique_frames = np.unique(frame_list)
                    n_frames = len(unique_frames)

                    channel_names = ["data"]
                    channel_list = [channel_names] * n_frames

                    channel_images = {}
                    for channel in channel_names:
                        if self.verbose:
                            print(f"Creating image memory for {dataset_name} {channel}...")

                        shared_image = self.create_import_shared_image(image_size)
                        channel_images[channel] = shared_image
                        shared_images[dataset_name][channel] = shared_image

                    image_dict = {"path": path,
                                  "dataset_name": dataset_name,
                                  "n_frames": n_frames,
                                  "channel_names": channel_names,
                                  "channel_list": channel_list,
                                  "frame_list": frame_list,
                                  "channel_frame_list": frame_list,
                                  "channel_images": channel_images,
                                  "image_shape": image_shape,
                                  "channel_layout": channel_layout,
                                  "alex_first_frame": alex_first_frame,
                                  "dtype": dtype,
                                  "import_mode": import_mode.lower()}

                    image_list.append(image_dict)

                elif import_mode.lower() == "fret":

                    if import_limit_combo != "None":
                        import_limit = int(self.gui.pixseq_import_limt.currentText())
                    else:
                        import_limit = n_frames

                    frame_list = list(range(n_frames))[:import_limit]

                    if channel_layout.lower() == "donor-acceptor":
                        channel_names = ["donor", "acceptor"]
                    else:
                        channel_names = ["acceptor", "donor"]

                    image_shape = (import_limit, image_shape[1], image_shape[2]//2)

                    unique_frames = np.unique(frame_list)
                    n_frames = len(unique_frames)

                    channel_list = [channel_names] * n_frames

                    channel_images = {}
                    for channel in channel_names:

                        if self.verbose:
                            print(f"Creating shared image for {dataset_name} {channel}...")

                        shared_image = self.create_import_shared_image(image_size//2)
                        channel_images[channel] = shared_image
                        shared_images[dataset_name][channel] = shared_image

                    image_dict = {"path": path,
                                  "dataset_name": dataset_name,
                                  "n_frames": n_frames,
                                  "channel_names": channel_names,
                                  "channel_list": channel_list,
                                  "frame_list": frame_list,
                                  "channel_frame_list": frame_list,
                                  "channel_images": channel_images,
                                  "image_shape": image_shape,
                                  "channel_layout": channel_layout,
                                  "alex_first_frame": alex_first_frame,
                                  "dtype": dtype,
                                  "import_mode": import_mode.lower()}

                    image_list.append(image_dict)

                elif import_mode.lower() == "alex":

                    if import_limit_combo != "None":
                        import_limit = int(self.gui.pixseq_import_limt.currentText())
                    else:
                        import_limit = n_frames

                    frame_list = list(range(n_frames))

                    even_frames = frame_list[::2][:import_limit]
                    odd_frames = frame_list[1::2][:import_limit]

                    frame_list = np.unique(np.concatenate([even_frames, odd_frames]))
                    frame_list = np.sort(frame_list).tolist()
                    channel_frame_list = np.repeat(np.arange(len(frame_list)//2), 2)

                    n_frames = len(frame_list)
                    image_shape = (n_frames//2, image_shape[1], image_shape[2]//2)

                    channel_list = []

                    for frame in frame_list:
                        if frame % 2 == 0:
                            if alex_first_frame.lower() == "donor":
                                channel_ex = "d"
                            else:
                                channel_ex = "a"
                        else:
                            if alex_first_frame.lower() == "donor":
                                channel_ex = "a"
                            else:
                                channel_ex = "d"

                        if channel_layout.lower() == "donor-acceptor":
                            channel_names = [f"{channel_ex}d", f"{channel_ex}a"]
                        else:
                            channel_names = [f"{channel_ex}a", f"{channel_ex}d"]

                        channel_list.append(channel_names)

                    channel_names = np.unique(channel_list)

                    channel_images = {}
                    for channel in channel_names:

                        if self.verbose:
                            print(f"Creating shared memory for {dataset_name} {channel}...")

                        shared_image = self.create_import_shared_image(image_size//4)
                        channel_images[channel] = shared_image
                        shared_images[dataset_name][channel] = shared_image

                    image_dict = {"path": path,
                                  "dataset_name": dataset_name,
                                  "n_frames": n_frames,
                                  "channel_names": channel_names,
                                  "channel_list": channel_list,
                                  "frame_list": frame_list,
                                  "channel_frame_list": channel_frame_list,
                                  "channel_images": channel_images,
                                  "image_shape": image_shape,
                                  "channel_layout": channel_layout,
                                  "alex_first_frame": alex_first_frame,
                                  "dtype": dtype,
                                  "import_mode": import_mode.lower()}

                    image_list.append(image_dict)

                channel_layout = self.gui.pixseq_channel_layout.currentText()
                alex_first_frame = self.gui.pixseq_alex_first_frame.currentText()

                if dataset_name not in import_dict.keys():
                    import_dict[dataset_name] = {"path":path,
                                                 "import_mode": import_mode.lower(),
                                                 "import_limit": import_limit,
                                                 "channel_layout": channel_layout,
                                                 "alex_first_frame": alex_first_frame,
                                                 "image_shape": image_shape,
                                                 "dtype": dtype,}

        except:
            print(traceback.format_exc())

        return image_list, shared_images, import_dict

    def populate_import_compute_jobs(self, image_list):

        if self.verbose:
            print(f"Populating import compute jobs.")

        compute_jobs = []

        for image_dict in image_list:

            frame_list = np.unique(image_dict["frame_list"])
            channel_list = image_dict["channel_list"]
            channel_frame_list = image_dict["channel_frame_list"]

            compute_jobs.append({"frame_list": frame_list,
                                 "channel_list": channel_list,
                                 "channel_frame_list": channel_frame_list,
                                 "stop_event": self.stop_event,
                                 **image_dict})

        return compute_jobs

    def process_compute_jobs(self, compute_jobs, progress_callback=None):

        if self.verbose:
            print(f"Processing {len(compute_jobs)} compute jobs.")

        cpu_count = int(multiprocessing.cpu_count() * 0.75)
        timeout_duration = 10  # Timeout in seconds

        with Manager() as manager:
            progress_dict = manager.dict()

            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:

                # Submit all jobs and store the future objects
                futures = [executor.submit(import_image_data, job, progress_dict, i) for i, job in enumerate(compute_jobs)]

                while any(not future.done() for future in futures):
                    # Calculate and emit progress
                    total_progress = sum(progress_dict.values())
                    overall_progress = int((total_progress / len(compute_jobs)))
                    if progress_callback is not None:
                        progress_callback.emit(overall_progress)
                    time.sleep(0.1)  # Update frequency

                # Wait for all futures to complete
                concurrent.futures.wait(futures)

                # Retrieve and process results
                results = [future.result() for future in futures]

        if self.verbose:
            print("Finished processing compute jobs.")

    def populate_import_dataset_dict(self, import_dict):

        try:

            if self.verbose:
                print("Populating dataset dict")

            for dataset_name, dataset_dict in import_dict.items():

                image_dict = {}

                path = dataset_dict["path"]
                image_shape = dataset_dict["image_shape"]
                dtype = dataset_dict["dtype"]
                import_mode = dataset_dict["import_mode"]
                channel_layout = dataset_dict["channel_layout"]
                alex_first_frame = dataset_dict["alex_first_frame"]

                dataset_images = self.shared_images[dataset_name]

                channel_names = dataset_images.keys()

                for channel_name, shared_mem in dataset_images.items():

                    image = np.ndarray(image_shape, dtype=dtype, buffer=shared_mem.buf).copy()
                    image = image.astype(np.uint16)

                    shared_mem.close()
                    shared_mem.unlink()

                    if channel_name not in image_dict.keys():
                        image_dict[channel_name] = {"data": []}

                    if channel_name.lower() in ["donor", "acceptor"]:
                        channel_display_name = channel_name.capitalize()
                    else:
                        channel_display_name = channel_name.upper()

                    if channel_name in ["donor", "acceptor", "da", "dd"]:
                        excitation = "d"
                    else:
                        excitation = "a"

                    if channel_name in ["donor", "ad", "dd"]:
                        emission = "d"
                    else:
                        emission = "a"

                    if import_mode.lower() == "fret":
                        fret = True
                    else:
                        fret = False

                    channel_ref = f"{excitation}{emission}"

                    image_dict[channel_name]["data"] = image
                    image_dict[channel_name]["path"] = path
                    image_dict[channel_name]["channel_ref"] = channel_ref
                    image_dict[channel_name]["excitation"] = excitation
                    image_dict[channel_name]["emission"] = emission
                    image_dict[channel_name]["channel_layout"] = channel_layout
                    image_dict[channel_name]["alex_first_frame"] = alex_first_frame
                    image_dict[channel_name]["FRET"] = fret
                    image_dict[channel_name]["import_mode"] = import_mode
                    image_dict[channel_name]["gap_label"] = None
                    image_dict[channel_name]["sequence_label"] = None

                if dataset_name not in self.dataset_dict.keys():
                    self.dataset_dict[dataset_name] = image_dict
                else:
                    for channel_name, channel_dict in image_dict.items():
                        self.dataset_dict[dataset_name][channel_name] = channel_dict

        except:
            pass

    def closed_import_shared_images(self):

        if hasattr(self, "shared_images"):

            if self.verbose:
                print("Closing import shared images.")

            for dataset_name, dataset_dict in self.shared_images.items():
                for channel_name, shared_mem in dataset_dict.items():
                    shared_mem.close()
                    shared_mem.unlink()

    def _pixseq_import_data(self, progress_callback=None, paths=[]):

        try:

            image_list, self.shared_images, import_dict = self.populate_import_lists(paths=paths)

            compute_jobs = self.populate_import_compute_jobs(image_list)

            self.process_compute_jobs(compute_jobs, progress_callback=progress_callback)

            self.populate_import_dataset_dict(import_dict)

            self.closed_import_shared_images()

        except:
            print(traceback.format_exc())
            self.update_ui()

    def populate_dataset_combos(self):

        try:

            if self.verbose:
                print("Populating all dataset combos.")

            dataset_names = list(self.dataset_dict.keys())

            self.gui.pixseq_dataset_selector.blockSignals(True)
            self.gui.pixseq_dataset_selector.clear()
            self.gui.pixseq_dataset_selector.addItems(dataset_names)
            self.gui.pixseq_dataset_selector.blockSignals(False)

            self.gui.cluster_dataset.blockSignals(True)
            self.gui.cluster_dataset.clear()
            self.gui.cluster_dataset.addItems(dataset_names)
            self.gui.cluster_dataset.blockSignals(False)

            self.gui.tform_compute_dataset.blockSignals(True)
            self.gui.tform_compute_dataset.clear()
            self.gui.tform_compute_dataset.addItems(dataset_names)
            self.gui.tform_compute_dataset.blockSignals(False)

            self.gui.pixseq_old_dataset_name.blockSignals(True)
            self.gui.pixseq_old_dataset_name.clear()
            self.gui.pixseq_old_dataset_name.addItems(dataset_names)
            self.gui.pixseq_old_dataset_name.blockSignals(False)

            self.gui.align_reference_dataset.blockSignals(True)
            self.gui.align_reference_dataset.clear()
            self.gui.align_reference_dataset.addItems(dataset_names)
            self.gui.align_reference_dataset.blockSignals(False)

            self.gui.colo_dataset.blockSignals(True)
            self.gui.colo_dataset.clear()
            self.gui.colo_dataset.addItems(dataset_names)
            self.gui.colo_dataset.blockSignals(False)

            self.gui.pixseq_append_dataset.blockSignals(True)
            self.gui.pixseq_append_dataset.clear()
            self.gui.pixseq_append_dataset.addItems(dataset_names)
            self.gui.pixseq_append_dataset.blockSignals(False)

            self.gui.delete_dataset_name.blockSignals(True)
            self.gui.delete_dataset_name.clear()
            self.gui.delete_dataset_name.addItems(dataset_names)
            self.gui.delete_dataset_name.blockSignals(False)

            self.gui.update_labels_dataset.blockSignals(True)
            self.gui.update_labels_dataset.clear()
            self.gui.update_labels_dataset.addItems(dataset_names)
            self.gui.update_labels_dataset.blockSignals(False)

            self.gui.import_picasso_dataset.blockSignals(True)
            self.gui.import_picasso_dataset.clear()
            self.gui.import_picasso_dataset.addItems(dataset_names)
            self.gui.import_picasso_dataset.blockSignals(False)

            self.gui.picasso_filter_dataset.blockSignals(True)
            self.gui.picasso_filter_dataset.clear()
            self.gui.picasso_filter_dataset.addItems(dataset_names)
            self.gui.picasso_filter_dataset.blockSignals(False)

            self.gui.tracking_dataset.blockSignals(True)
            self.gui.tracking_dataset.clear()
            self.gui.tracking_dataset.addItems(dataset_names)
            self.gui.tracking_dataset.blockSignals(False)

            self.gui.cellpose_dataset.blockSignals(True)
            self.gui.cellpose_dataset.clear()
            self.gui.cellpose_dataset.addItems(dataset_names)
            self.gui.cellpose_dataset.blockSignals(False)

            if len(dataset_names) > 1:
                dataset_names.insert(0, "All Datasets")

            self.gui.traces_export_dataset.blockSignals(True)
            self.gui.traces_export_dataset.clear()
            self.gui.traces_export_dataset.addItems(dataset_names)
            self.gui.traces_export_dataset.blockSignals(False)

            self.gui.filtering_datasets.blockSignals(True)
            self.gui.filtering_datasets.clear()
            self.gui.filtering_datasets.addItems(dataset_names)
            self.gui.filtering_datasets.blockSignals(False)

            self.gui.picasso_dataset.blockSignals(True)
            self.gui.picasso_dataset.clear()
            self.gui.picasso_dataset.addItems(dataset_names)
            self.gui.picasso_dataset.blockSignals(False)

            self.gui.undrift_dataset_selector.blockSignals(True)
            self.gui.undrift_dataset_selector.clear()
            self.gui.undrift_dataset_selector.addItems(dataset_names)
            self.gui.undrift_dataset_selector.blockSignals(False)

            self.gui.export_dataset.blockSignals(True)
            self.gui.export_dataset.clear()
            self.gui.export_dataset.addItems(dataset_names)
            self.gui.export_dataset.blockSignals(False)

            self.gui.locs_export_dataset.blockSignals(True)
            self.gui.locs_export_dataset.clear()
            self.gui.locs_export_dataset.addItems(dataset_names)
            self.gui.locs_export_dataset.blockSignals(False)

            self.gui.simple_plot_dataset.blockSignals(True)
            self.gui.simple_plot_dataset.clear()
            self.gui.simple_plot_dataset.addItems(dataset_names)
            self.gui.simple_plot_dataset.blockSignals(False)

            self.gui.picasso_render_dataset.blockSignals(True)
            self.gui.picasso_render_dataset.clear()
            self.gui.picasso_render_dataset.addItems(dataset_names)
            self.gui.picasso_render_dataset.blockSignals(False)

        except:
            print(traceback.format_exc())

    def initialise_localisation_dict(self):

        if hasattr(self, "localisation_dict"):
            self.localisation_dict = {"bounding_boxes": {}, "localisations": {}}

        if hasattr(self, "dataset_dict"):

            if self.verbose:
                print("Initialising localisation dict.")

            for dataset_name, dataset_dict in self.dataset_dict.items():

                if dataset_name not in self.localisation_dict.keys():
                    self.localisation_dict["localisations"][dataset_name] = {}

                fiducial_dict = self.localisation_dict["localisations"][dataset_name]

                for channel_name, channel_dict in dataset_dict.items():
                    if channel_name not in fiducial_dict.keys():
                        fiducial_dict[channel_name.lower()] = {}

                self.localisation_dict["localisations"][dataset_name] = fiducial_dict

    def _pixseq_import_data_finished(self):

        if self.verbose:
            print("Finished importing data, executing post import functions")

        self.initialise_localisation_dict()
        self.populate_dataset_combos()

        self.update_channel_select_buttons()
        self.populate_channel_selectors()
        self.update_active_image()
        self.update_export_options()
        self.populate_export_combos()
        self.update_filtering_channels()
        self.update_loc_export_options()

        self.update_align_reference_channel()

        self.update_ui()

    def pixseq_import_data(self):

        try:

            append_dataset = self.gui.pixseq_append_dataset.currentText()

            if self.gui.pixseq_append.isChecked() and append_dataset not in self.dataset_dict.keys():
                print("Please select a dataset to append to")
            else:

                desktop = os.path.expanduser("~/Desktop")
                paths = QFileDialog.getOpenFileNames(self, 'Open file', desktop, "Image files (*.tif *.tiff)")[0]

                paths = [path for path in paths if path != ""]

                if paths != []:

                    self.update_ui(init=True)

                    self.worker = Worker(self._pixseq_import_data, paths=paths)
                    self.worker.signals.progress.connect(partial(self.pixseq_progress,
                        progress_bar=self.gui.pixseq_import_progressbar))
                    self.worker.signals.finished.connect(self._pixseq_import_data_finished)
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)

        except:
            self.update_ui()
            pass