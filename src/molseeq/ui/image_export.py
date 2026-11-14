import os
import traceback
from concurrent.futures import CancelledError
from contextlib import contextmanager

import numpy as np
import tifffile
from molcore.files import atomic_files
from qtpy.QtWidgets import QFileDialog

from molseeq.ui.worker import Worker


def write_image_jobs(jobs, stop_event, progress_callback):
    paths = [path for path, *_ in jobs]
    total = sum(shape[0] for _, shape, _, _ in jobs)
    if not paths or not total:
        return

    def check():
        if stop_event.is_set():
            raise CancelledError("Image export cancelled")

    @contextmanager
    def publish_guard():
        check()
        yield

    done = 0
    with atomic_files(paths, publish_guard=publish_guard) as staged_paths:
        for (_, shape, dtype, frames), staged in zip(jobs, staged_paths, strict=True):
            size = int(np.prod(shape)) * dtype.itemsize
            with tifffile.TiffWriter(staged, bigtiff=size >= 2**32) as writer:
                for frame in frames:
                    check()
                    writer.write(frame, photometric="minisblack", contiguous=True, metadata=None)
                    done += 1
                    progress_callback.emit(done * 100 // total)
        if done != total:
            raise ValueError("Image source ended before the expected frame count")


class ImageExportMixin:

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
                file_name = file_name.replace("_molseeq_processed","")

            else:

                paths = [channel["path"] for channel in dataset_dict.values()]
                file_name = os.path.basename(paths[0])
                export_dir = os.path.dirname(paths[0])
                file_name, file_extension = os.path.splitext(file_name)
                file_name = file_name.replace("_molseeq_processed", "")

            if export_channel.lower() not in ["alex", "fret"]:
                name_modifier = f"_{export_channel}_molseeq_processed.tif"
                if name_modifier not in file_name:
                    export_path = os.path.join(export_dir,file_name + name_modifier)
                else:
                    export_path = os.path.join(export_dir,file_name,".tif")
            else:
                name_modifier = "_molseeq_processed.tif"
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
            file_name = file_name.replace("_molseeq_processed", "")

            if export_channel.lower() not in ["alex", "fret", "single channel"]:
                name_modifier = f"_{export_channel}_molseeq_processed.tif"
                if name_modifier not in file_name:
                    export_path = os.path.join(export_dir, file_name + name_modifier)
                else:
                    export_path = os.path.join(export_dir, file_name, ".tif")
            else:
                name_modifier = "_molseeq_processed.tif"
                if name_modifier not in file_name:
                    export_path = os.path.join(export_dir, file_name + name_modifier)
                else:
                    export_path = os.path.join(export_dir, file_name, ".tif")

            export_path = os.path.normpath(export_path)
            export_dict["export_path"] = export_path

            export_jobs[index] = export_dict


        return export_jobs, total_frames

    def export_data(self):
        if not self.session.images:
            return
        try:
            requests, _ = self.get_export_jobs()
            snapshot = self.session.image_snapshot()
            jobs = []
            for request in requests:
                name = request["dataset_name"]
                channel = request["export_channel"].lower()
                path = request["export_path"]
                if channel in ("fret", "alex"):
                    component = "dd" if channel == "alex" else "donor"
                    view = snapshot.channel(name, component)
                    count = len(view)
                    if channel == "alex":
                        count += len(snapshot.channel(name, "aa"))
                    shape = (count, view.shape[1], view.shape[2] * 2)
                    jobs.append((path, shape, view.dtype, snapshot.camera_frames(name, channel)))
                else:
                    source = snapshot.working_source(name, "data" if channel == "single channel" else channel)
                    jobs.append((path, source.shape, source.dtype, iter(source)))
            if not jobs:
                return
            self.update_ui(init=True)
            self.worker = Worker(write_image_jobs, jobs, self.stop_event)
            self.worker.signals.progress.connect(self.gui.export_progressbar.setValue)
            self.worker.signals.error.connect(lambda error: self.molseeq_notification(str(error[1])))
            self.worker.signals.finished.connect(self.update_ui)
            self.threadpool.start(self.worker)
        except (ValueError, KeyError) as error:
            self.molseeq_notification(str(error))
            self.update_ui()
