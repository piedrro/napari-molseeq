import traceback
import numpy as np
import pandas as pd

from molseeq.funcs.utils_compute import Worker
import scipy.ndimage
import multiprocessing
from multiprocessing import shared_memory
from functools import partial
import concurrent.futures
from picasso.postprocess import undrift as picasso_undrift
from picasso.aim import aim
from multiprocessing import Manager
import time


def undrift_image(dat):

    try:

        drift = dat["drift"]
        frame_index = dat["frame_index"]
        stop_event = dat["stop_event"]

        if not stop_event.is_set():

            # Access the shared memory
            shared_mem = shared_memory.SharedMemory(name=dat["shared_memory_name"])
            np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)

            # Perform preprocessing steps and overwrite original image
            img = np_array[frame_index]

            drift = [-drift[1], -drift[0]]

            img = scipy.ndimage.shift(img, drift, mode='constant', cval=0.0)

            # overwrite the shared memory block
            np_array[frame_index] = img

    except:
        print(traceback.format_exc())
        pass

    return frame_index


def detect_aim_dataset_drift(dat, progress_dict, index):

    dataset_dict = dat.get("dataset_dict", {})
    segmentation = dat.get("segmentation")
    intersect_d = dat.get("intersect_d")
    roi_r = dat.get("roi_r")

    try:
        loc_dict = dataset_dict["loc_dict"]
        undrift_locs = loc_dict["localisations"].copy()
        picasso_info = dataset_dict["picasso_info"]
        n_frames = picasso_info[0]["Frames"]

        undrift_locs = pd.DataFrame(undrift_locs)

        if "dataset" in undrift_locs.columns:
            undrift_locs = undrift_locs.drop(columns=["dataset"])
        if "channel" in undrift_locs.columns:
            undrift_locs = undrift_locs.drop(columns=["channel"])

        undrift_locs = undrift_locs.to_records(index=False)

        max_iter = (n_frames + segmentation - 1) // segmentation * 2 - 1

        class AIMProgress:
            def __init__(self):
                self.iter = 0
            def set_value(self, value):
                self.iter += 1
                progress = int(self.iter/max_iter * 50)
                progress_dict[index] = progress
            def zero_progress(self, description=None):
                pass
            def close(self):
                pass

        aim_progress = AIMProgress()

        if type(segmentation) == int:
            if n_frames > segmentation:
                undrifted_locs, new_info, drift = aim(
                    locs = undrift_locs,
                    info=picasso_info,
                    segmentation=segmentation,
                    intersect_d=intersect_d,
                    roi_r=roi_r,
                    progress=aim_progress,
                    )
                dataset_dict["drift"] = drift
                dataset_dict["undrifted_locs"] = undrifted_locs
            else:
                progress_dict[index] = 100
        else:
            progress_dict[index] = 100
    except:
        print(traceback.format_exc())
        pass

    return dataset_dict

def detect_rcc_dataset_drift(dat, progress_dict, index):

    dataset_dict = dat.get("dataset_dict", {})
    segmentation = dat.get("segmentation")


    try:

        loc_dict = dataset_dict["loc_dict"]
        undrift_locs = loc_dict["localisations"].copy()
        picasso_info = dataset_dict["picasso_info"]
        n_frames = picasso_info[0]["Frames"]

        undrift_locs = pd.DataFrame(undrift_locs)

        if "dataset" in undrift_locs.columns:
            undrift_locs = undrift_locs.drop(columns=["dataset"])
        if "channel" in undrift_locs.columns:
            undrift_locs = undrift_locs.drop(columns=["channel"])

        undrift_locs = undrift_locs.to_records(index=False)

        len_segments = n_frames // segmentation
        n_pairs = int(len_segments * (len_segments - 1))/2

        compute_progress = {"segmentation": 0,"undrift": 0}

        def total_progress():
            segmentation_progress = compute_progress["segmentation"]
            undrift_progress = compute_progress["undrift"]
            total_progress = int((segmentation_progress + undrift_progress)/2)
            progress_dict[index] = total_progress
        def segmentation_callback(progress):
            compute_progress["segmentation"] = (progress/len_segments)*100
            total_progress()

        def undrift_callback(progress):
            compute_progress["undrift"] = (progress/n_pairs)*100
            total_progress()

        if type(segmentation) == int:
            if n_frames > segmentation:
                drift, undrifted_locs = picasso_undrift(undrift_locs,
                    picasso_info,
                    segmentation=segmentation,
                    display=False,
                    segmentation_callback=segmentation_callback,
                    rcc_callback=undrift_callback,
                    )
                dataset_dict["drift"] = drift
                dataset_dict["undrifted_locs"] = undrifted_locs
            else:
                progress_dict[index] = 100
        else:
            progress_dict[index] = 100

    except:
        print(traceback.format_exc())
        pass

    return dataset_dict


class _undrift_utils:

    def undrift_localisations(self):

        try:

            for dataset_name, dataset_data in self.dataset_dict.items():
                for channel_name, channel_data in self.dataset_dict[dataset_name].items():
                    fiducial_dict = self.localisation_dict["localisations"][dataset_name][channel_name.lower()].copy()

                    if "drift" in channel_data.keys() and "localisations" in fiducial_dict.keys():
                        locs = fiducial_dict["localisations"]

                        drift = channel_data["drift"]

                        for loc in locs:
                            loc.x = loc.x - drift[loc.frame][0]
                            loc.y = loc.y - drift[loc.frame][1]

                        self.localisation_dict["localisations"][dataset_name][channel_name.lower()]["localisations"] = locs

        except:
            print(traceback.format_exc())
            pass

    def _undrift_images_finished(self):

        try:

            self.image_layer.data = self.dataset_dict[self.active_dataset][self.active_channel]["data"]

            self.undrift_localisations()
            self.draw_localisations(update_vis=True)

            for layer in self.viewer.layers:
                layer.refresh()

            self.update_ui()

        except:
            print(traceback.format_exc())
            self.update_ui()

    def _apply_undrift(self, progress_callback, undrift_dict):

        try:

            if undrift_dict != None:

                self.shared_images = self.create_shared_images()

                compute_jobs = []

                for image_dict in self.shared_images:

                    dataset = image_dict["dataset"]
                    channel = image_dict["channel"]
                    n_frames = image_dict['shape'][0]

                    frame_index_list = list(range(n_frames))

                    if "drift" in undrift_dict[dataset].keys():

                        image_drift = undrift_dict[dataset]["drift"]

                        self.dataset_dict[dataset][channel.lower()]["drift"] = image_drift

                        for frame_index, frame_drift in zip(frame_index_list, image_drift):

                            compute_jobs.append({"shared_memory_name": image_dict["shared_memory_name"],
                                                 "shape": image_dict["shape"],
                                                 "dtype": image_dict["dtype"],
                                                 "frame_index": frame_index,
                                                 "drift": frame_drift,
                                                 "stop_event": self.stop_event,
                                                 })

                cpu_count = int(multiprocessing.cpu_count() * 0.9)
                timeout_duration = 10  # Timeout in seconds

                with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                    # Submit all jobs and store the future objects
                    futures = {executor.submit(undrift_image, job): job for job in compute_jobs}

                    iter = 0
                    for future in concurrent.futures.as_completed(futures):

                        if self.stop_event.is_set():
                            future.cancel()
                        else:
                            job = futures[future]
                            try:
                                result = future.result(timeout=timeout_duration)  # Process result here
                            except concurrent.futures.TimeoutError:
                                # print(f"Task {job} timed out after {timeout_duration} seconds.")
                                pass
                            except Exception as e:
                                # print(f"Error occurred in task {job}: {e}")  # Handle other exceptions
                                pass

                            # Update progress
                            iter += 1
                            progress = 50 + int((iter / len(compute_jobs)) * 50)
                            progress_callback.emit(progress)  # Emit the signal

                self.restore_shared_images()

        except:
            self.restore_shared_images()

            self.update_ui()

            print(traceback.format_exc())
            pass


    def _detect_undrift(self, progress_callback, undrift_dict, **kwargs):

        try:

            mode = kwargs.get('mode', 'RCC')

            if mode == 'RCC':
                detect_func = detect_rcc_dataset_drift
            else:
                detect_func = detect_aim_dataset_drift

            if undrift_dict != {}:
                compute_jobs = []
                progress_dict = {}

                for dataset, dataset_dict in undrift_dict.items():

                    job = {**{"dataset": dataset, "dataset_dict": dataset_dict}, **kwargs}

                    compute_jobs.append(job)
                    progress_dict[dataset] = 0

                cpu_count = int(multiprocessing.cpu_count() * 0.9)

                with Manager() as manager:
                    progress_dict = manager.dict()

                    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                        # Submit all jobs
                        futures = [executor.submit(detect_func, job, progress_dict, i) for i, job in enumerate(compute_jobs)]

                        while any(not future.done() for future in futures):
                            # Calculate and emit progress
                            total_progress = sum(progress_dict.values())
                            overall_progress = int((total_progress / len(compute_jobs))/2)
                            if progress_callback is not None:
                                progress_callback.emit(overall_progress)
                            time.sleep(1)  # Update frequency

                        # Wait for all futures to complete
                        concurrent.futures.wait(futures)

                        # Retrieve and process results
                        results = [future.result() for future in futures]
                        for result in results:
                            if result is not None:
                                if "drift" in result.keys():
                                    drift = result["drift"]
                                    dataset = result["dataset"]
                                    undrift_dict[dataset]["drift"] = drift

        except:
            print(traceback.format_exc())
            pass

        return undrift_dict


    def _undrift(self, progress_callback=None, undrift_dict={}, **kwargs):

        try:

           undrift_dict = self._detect_undrift(progress_callback, undrift_dict, **kwargs)
           self._apply_undrift(progress_callback, undrift_dict)

        except:
            print(traceback.format_exc())
            pass


    def aim_undrift(self):

        try:

            dataset = self.gui.undrift_dataset_selector.currentText()
            channel = self.gui.undrift_channel_selector.currentText()
            segmentation = self.gui.aim_segmentation.value()
            intersect_d = self.gui.aim_intersect_d.value()
            roi_r = self.gui.aim_roi_r.value()

            if dataset == "All Datasets":
                dataset_list = list(self.dataset_dict.keys())
            else:
                dataset_list = [dataset]

            undrift_dict = {}

            for dataset in dataset_list:
                loc_dict, n_locs, _ = self.get_loc_dict(dataset, channel.lower())
                if n_locs > 0 and loc_dict["fitted"] == True:

                    n_frames, height, width = self.dataset_dict[dataset][channel.lower()]["data"].shape
                    pixel_size = self.dataset_dict[dataset][channel.lower()]["pixel_size"]
                    picasso_info = [{'Frames': n_frames, 'Height': height,
                                     'Width': width, 'Pixelsize': pixel_size}, {}]
                    undrift_dict[dataset] = {"loc_dict": loc_dict, "n_locs": n_locs,
                                             "picasso_info": picasso_info,
                                             "channel": channel.lower(), "dataset": dataset}
                else:
                    self.molseeq_notification("No fitted localizations found for dataset: " + dataset)

            if undrift_dict != {}:
                self.update_ui(init=True)

                self.worker = Worker(self._undrift,
                                     mode = 'AIM',
                                     undrift_dict=undrift_dict,
                                     segmentation=segmentation,
                                     intersect_d = intersect_d,
                                     roi_r = roi_r)
                self.worker.signals.progress.connect(
                    partial(self.molseeq_progress, progress_bar=self.gui.aim_progressbar))
                self.worker.signals.finished.connect(self._undrift_images_finished)
                self.threadpool.start(self.worker)

        except:
            self.update_ui()
            print(traceback.format_exc())
            pass



    def rcc_undrift(self):

        try:

            dataset = self.gui.undrift_dataset_selector.currentText()
            channel = self.gui.undrift_channel_selector.currentText()
            segmentation = self.gui.undrift_segmentation.value()

            if dataset == "All Datasets":
                dataset_list = list(self.dataset_dict.keys())
            else:
                dataset_list = [dataset]

            undrift_dict = {}

            for dataset in dataset_list:
                loc_dict, n_locs, _ = self.get_loc_dict(dataset, channel.lower())
                if n_locs > 0 and loc_dict["fitted"] == True:

                    n_frames,height,width = self.dataset_dict[dataset][channel.lower()]["data"].shape
                    picasso_info = [{'Frames': n_frames, 'Height': height, 'Width': width}, {}]

                    undrift_dict[dataset] = {"loc_dict": loc_dict, "n_locs": n_locs,
                                             "picasso_info": picasso_info,
                                             "channel": channel.lower(), "dataset": dataset}
                else:
                    self.molseeq_notification("No fitted localizations found for dataset: " + dataset)

            if undrift_dict != {}:

                self.update_ui(init=True)

                self.worker = Worker(self._undrift,
                                     mode = 'RCC',
                                     undrift_dict=undrift_dict,
                                     segmentation=segmentation)
                self.worker.signals.progress.connect(partial(self.molseeq_progress, progress_bar=self.gui.undrift_progressbar))
                self.worker.signals.finished.connect(self._undrift_images_finished)
                self.threadpool.start(self.worker)

        except:
            self.update_ui()
            print(traceback.format_exc())
            pass

