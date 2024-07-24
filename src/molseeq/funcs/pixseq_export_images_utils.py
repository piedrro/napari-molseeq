import numpy as np
import traceback
import tifffile
import os
import psutil
from qtpy.QtWidgets import QFileDialog
from molseeq.funcs.pixseq_utils_compute import Worker
from functools import partial

class _export_images_utils:

    def common_elements(self, list_of_lists):

        common_set = set(list_of_lists[0])

        for sublist in list_of_lists[1:]:
            common_set = common_set.intersection(sublist)

        return list(common_set)

    def update_export_options(self):

        if self.dataset_dict != {}:

            dataset_name = self.gui.export_dataset.currentText()

            export_channel_list = []

            if dataset_name == "All Datasets":
                dataset_list = list(self.dataset_dict.keys())
            else:
                dataset_list = [dataset_name]

            for dataset_name in dataset_list:

                import_mode_list = []

                dataset_channel_list = []

                for channel_name, channel_data in self.dataset_dict[dataset_name].items():

                    import_mode_list.append(channel_data["import_mode"])

                    if channel_name.lower() in ["donor", "acceptor", "data"]:
                        channel_name = channel_name.capitalize()
                    else:
                        channel_name = channel_name.upper()

                    dataset_channel_list.append(channel_name)
                export_channel_list.append(dataset_channel_list)

            export_channel_list = self.common_elements(export_channel_list)

            export_channel_list.insert(0, "Import Channel(s)")

            self.gui.export_channel.clear()
            self.gui.export_channel.addItems(export_channel_list)

    def get_free_RAM(self):

            try:

                memory_info = psutil.virtual_memory()
                available_memory = memory_info.available

                return available_memory

            except:
                print(traceback.format_exc())
                pass

    def get_export_path(self,dialog=False):

        export_path = None

        try:

            dataset_name = self.gui.export_dataset.currentText()
            export_channel = self.gui.export_channel.currentText()

            dataset_dict = self.dataset_dict[dataset_name]

            if export_channel.lower() not in ["alex", "fret"]:

                channel_dict = dataset_dict[export_channel.lower()]
                image_path = channel_dict["path"]

                export_path = os.path.normpath(image_path)
                export_dir = os.path.dirname(export_path)
                file_name = os.path.basename(export_path)
                file_name = os.path.splitext(file_name)[0]
                file_name, file_extension = os.path.splitext(file_name)
                file_name = file_name.replace("_pixseq_processed","")

            else:

                paths = [channel["path"] for channel in dataset_dict.values()]
                file_name = os.path.basename(paths[0])
                export_dir = os.path.dirname(paths[0])
                file_name, file_extension = os.path.splitext(file_name)
                file_name = file_name.replace("_pixseq_processed", "")

            if export_channel.lower() not in ["alex", "fret"]:
                name_modifier = f"_{export_channel}_pixseq_processed.tif"
                if name_modifier not in file_name:
                    export_path = os.path.join(export_dir,file_name + name_modifier)
                else:
                    export_path = os.path.join(export_dir,file_name,".tif")
            else:
                name_modifier = "_pixseq_processed.tif"
                if name_modifier not in file_name:
                    export_path = os.path.join(export_dir,file_name + name_modifier)
                else:
                    export_path = os.path.join(export_dir,file_name,".tif")

            if dialog == True:
                export_path = QFileDialog.getSaveFileName(self, 'Save ALEX data', export_path, 'Text files (*.tif)')[0]

            export_path = os.path.normpath(export_path)

        except:
            print(traceback.format_exc())
            pass

        return export_path



    def get_export_jobs(self):

        dataset_name = self.gui.export_dataset.currentText()
        export_channel = self.gui.export_channel.currentText()

        export_jobs = []
        total_frames = 0

        if dataset_name == "All Datasets":

            for dataset_name in list(self.dataset_dict.keys()):
                dataset_dict = self.dataset_dict[dataset_name]

                if export_channel != "Import Channel(s)":

                    path = dataset_dict[export_channel.lower()]["path"]

                    export_jobs.append({"dataset_name": dataset_name,
                                        "export_channel": export_channel,
                                        "import_path": path})

                    n_frames = dataset_dict[export_channel.lower()]["data"].shape[0]
                    total_frames += n_frames

                else:
                    import_modes = np.unique([channel_dict["import_mode"] for channel_dict in dataset_dict.values()])
                    channel_names = np.unique([channel_name for channel_name in dataset_dict.keys()])
                    import_paths = np.unique([channel_dict["path"] for channel_dict in dataset_dict.values()])

                    for path, mode, channel in zip(import_paths, import_modes, channel_names):

                        export_jobs.append({"dataset_name": dataset_name,
                                            "export_channel": mode.upper(),
                                            "import_path": path})

                        n_frames = dataset_dict[channel]["data"].shape[0]
                        total_frames += n_frames

        else:

            dataset_dict = self.dataset_dict[dataset_name]

            if export_channel != "Import Channel(s)":

                path = dataset_dict[export_channel.lower()]["path"]

                export_jobs.append({"dataset_name": dataset_name,
                                    "export_channel": export_channel,
                                    "import_path": path})

                n_frames = dataset_dict[export_channel.lower()]["data"].shape[0]
                total_frames += n_frames

            else:
                import_modes = np.unique([channel_dict["import_mode"] for channel_dict in dataset_dict.values()])
                channel_names = np.unique([channel_name for channel_name in dataset_dict.keys()])
                import_paths = np.unique([channel_dict["path"] for channel_dict in dataset_dict.values()])

                for path, mode, channel in zip(import_paths, import_modes, channel_names):

                    export_jobs.append({"dataset_name": dataset_name,
                                        "export_channel": mode.upper(),
                                        "import_path": path})

                    n_frames = dataset_dict[channel]["data"].shape[0]
                    total_frames += n_frames

        for index, export_dict in enumerate(export_jobs):

            import_path = export_dict["import_path"]
            export_channel = export_dict["export_channel"]

            export_path = os.path.normpath(import_path)
            export_dir = os.path.dirname(export_path)
            file_name = os.path.basename(export_path)
            file_name = os.path.splitext(file_name)[0]
            file_name, file_extension = os.path.splitext(file_name)
            file_name = file_name.replace("_pixseq_processed", "")

            if export_channel.lower() not in ["alex", "fret", "single channel"]:
                name_modifier = f"_{export_channel}_pixseq_processed.tif"
                if name_modifier not in file_name:
                    export_path = os.path.join(export_dir, file_name + name_modifier)
                else:
                    export_path = os.path.join(export_dir, file_name, ".tif")
            else:
                name_modifier = "_pixseq_processed.tif"
                if name_modifier not in file_name:
                    export_path = os.path.join(export_dir, file_name + name_modifier)
                else:
                    export_path = os.path.join(export_dir, file_name, ".tif")

            export_path = os.path.normpath(export_path)
            export_dict["export_path"] = export_path

            export_jobs[index] = export_dict


        return export_jobs, total_frames

    def export_data_finished(self):

        self.update_ui()


    def export_data(self):

        try:

            if self.dataset_dict != {}:

                export_jobs, total_frames = self.get_export_jobs()

                progress_dict = {}

                self.pixseq_notification(f"Exporting image data...")

                for job_index, export_jobs in enumerate(export_jobs):

                    if job_index not in progress_dict.keys():
                        progress_dict[job_index] = 0

                    dataset_name = export_jobs["dataset_name"]
                    export_channel = export_jobs["export_channel"]
                    export_path = export_jobs["export_path"]

                    self.update_ui(init=True)

                    def export_progress(progress, job_index=None):
                        progress_dict[job_index] = progress
                        total_progress = int(np.sum(list(progress_dict.values()))/len(progress_dict))
                        self.pixseq_progress(total_progress, self.gui.export_progressbar)

                    if export_channel.lower() == "alex":
                        self.worker = Worker(self.export_alex_data,
                            dataset_name=dataset_name,
                            export_path=export_path)
                        self.worker.signals.progress.connect(partial(export_progress, job_index=job_index))
                        self.worker.signals.finished.connect(self.export_data_finished)
                        self.worker.signals.error.connect(self.update_ui)
                        self.threadpool.start(self.worker)

                    elif export_channel.lower() == "fret":
                        self.worker = Worker(self.export_fret_data,
                            dataset_name=dataset_name,
                            export_path=export_path)
                        self.worker.signals.progress.connect(partial(export_progress, job_index=job_index))
                        self.worker.signals.finished.connect(self.export_data_finished)
                        self.worker.signals.error.connect(self.update_ui)
                        self.threadpool.start(self.worker)

                    elif export_channel.lower() == "single channel":
                        self.worker = Worker(self.export_channel_data,
                            dataset_name=dataset_name,
                            export_channel="Data",
                            export_path=export_path)
                        self.worker.signals.progress.connect(partial(export_progress, job_index=job_index))
                        self.worker.signals.finished.connect(self.export_data_finished)
                        self.worker.signals.error.connect(self.update_ui)
                        self.threadpool.start(self.worker)

                    else:
                        self.worker = Worker(self.export_channel_data,
                            dataset_name=dataset_name,
                            export_channel=export_channel,
                            export_path=export_path)
                        self.worker.signals.progress.connect(partial(export_progress, job_index=job_index))
                        self.worker.signals.finished.connect(self.export_data_finished)
                        self.worker.signals.error.connect(self.update_ui)
                        self.threadpool.start(self.worker)

        except:
            print(traceback.format_exc())
            pass

    def export_channel_data(self, progress_callback = None, dataset_name="", export_channel="", export_path=""):

        try:

            channel_dict = self.dataset_dict[dataset_name][export_channel.lower()]

            image = channel_dict["data"]

            tifffile.imwrite(export_path, image)

            self.pixseq_notification(f"Exported {export_channel} data to {export_path}")

        except:
            print(traceback.format_exc())
            pass

    def export_fret_data(self, progress_callback = None, dataset_name="", export_path=""):

        try:

            dataset_dict = self.dataset_dict[dataset_name]

            image_shapes = [channel_data["data"].shape for channel_name, channel_data in dataset_dict.items()]
            image_disk_sizes = [abs(int(channel_data["data"].nbytes)) for channel_name, channel_data in dataset_dict.items()]

            export_dir = os.path.dirname(export_path)

            if export_path != "" and os.path.isdir(export_dir):

                image_disk_size = 0
                for size in image_disk_sizes:
                    image_disk_size += size

                free_RAM = self.get_free_RAM()

                if free_RAM > image_disk_size:
                    disk_export = False
                else:
                    disk_export = True

                n_frames = image_shapes[0][0]
                height = image_shapes[0][1]
                width = image_shapes[0][2]

                export_shape = (n_frames,height,width*2)

                if disk_export == True:
                    disk_array_path = 'mmapped_array.dat'
                    export_array = np.memmap(disk_array_path, dtype='uint16', mode='w+', shape=export_shape)
                else:
                    disk_array_path = None
                    export_array = np.zeros(export_shape, dtype="uint16")

                iter = 0

                n_frames = n_frames*len(dataset_dict.keys())

                for channel_name, channel_data in dataset_dict.items():

                    channel_layout = channel_data["channel_layout"]
                    alex_first_frame = channel_data["alex_first_frame"]
                    channel_ref = channel_data["channel_ref"]

                    for frame_index in range(len(channel_data["data"])):

                        frame = channel_data["data"][frame_index]

                        left_image = False
                        if channel_layout == "Donor-Acceptor":
                            if channel_ref[-1] == "d":
                                left_image = True
                        else:
                            if channel_ref[-1] == "a":
                                left_image = True

                        if left_image == True:
                            export_array[frame_index][0:height,0:width] = frame
                        else:
                            export_array[frame_index][0:height,width:width*2] = frame

                        iter += 1

                        if progress_callback != None:
                            progress = int(iter/n_frames*100)
                            progress_callback.emit(progress)

                if disk_export:
                    # Make sure to flush changes to disk
                    export_array.flush()

                    with tifffile.TiffWriter(export_path) as tiff:
                        for idx in range(export_array.shape[0]):
                            tiff.write(export_array[idx])

                    if os.path.exists(disk_array_path):
                        os.remove(disk_array_path)

                else:
                    tifffile.imwrite(export_path, export_array)
                    del export_array

                print(f"Exported FRET data to {export_path}")

        except:
            pass


    def export_alex_data(self, progress_callback=None, dataset_name = "", export_path = ""):

        try:

            dataset_dict = self.dataset_dict[dataset_name]

            image_shapes = [channel_data["data"].shape for channel_name, channel_data in dataset_dict.items()]
            image_disk_sizes = [abs(int(channel_data["data"].nbytes)) for channel_name, channel_data in dataset_dict.items()]

            export_dir = os.path.dirname(export_path)

            if export_path != "" and os.path.isdir(export_dir):

                print("Exporting ALEX data")

                image_disk_size = 0
                for size in image_disk_sizes:
                    image_disk_size += size

                free_RAM = self.get_free_RAM()

                if free_RAM > image_disk_size:
                    disk_export = False
                else:
                    disk_export = True

                n_frames = image_shapes[0][0]
                height = image_shapes[0][1]
                width = image_shapes[0][2]

                export_shape = (n_frames*2,height,width*2)

                if disk_export == True:
                    disk_array_path = 'mmapped_array.dat'
                    export_array = np.memmap(disk_array_path, dtype='uint16', mode='w+', shape=export_shape)
                else:
                    disk_array_path = None
                    export_array = np.zeros(export_shape, dtype="uint16")

                iter = 0

                n_frames = n_frames*len(dataset_dict.keys())

                for channel_name, channel_data in dataset_dict.items():

                    channel_layout = channel_data["channel_layout"]
                    alex_first_frame = channel_data["alex_first_frame"]
                    channel_ref = channel_data["channel_ref"]

                    for frame_index in range(len(channel_data["data"])):

                        frame = channel_data["data"][frame_index]

                        left_image = False
                        if channel_layout.lower() == "donor-acceptor":
                            if channel_ref[-1] == "d":
                                left_image = True
                        else:
                            if channel_ref[-1] == "a":
                                left_image = True

                        if alex_first_frame == "Donor":
                            if channel_ref[0] == "d":
                                mapped_frame_index = frame_index * 2
                            else:
                                mapped_frame_index = frame_index * 2 + 1
                        else:
                            if channel_ref[0] == "a":
                                mapped_frame_index = frame_index * 2
                            else:
                                mapped_frame_index = frame_index * 2 + 1

                        if left_image == True:
                            export_array[mapped_frame_index][0:height,0:width] = frame
                        else:
                            export_array[mapped_frame_index][0:height,width:width*2] = frame

                        iter += 1

                        if progress_callback != None:
                            progress = int(iter/n_frames*100)
                            progress_callback.emit(progress)

                if disk_export:
                    # Make sure to flush changes to disk
                    export_array.flush()

                    with tifffile.TiffWriter(export_path) as tiff:
                        for idx in range(export_array.shape[0]):
                            tiff.write(export_array[idx])

                    if os.path.exists(disk_array_path):
                        os.remove(disk_array_path)

                else:
                    tifffile.imwrite(export_path, export_array)
                    del export_array

                print(f"Exported ALEX data to {export_path}")

        except:
            print(traceback.format_exc())
            return None