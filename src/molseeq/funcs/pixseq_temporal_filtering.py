import numpy as np
from numba import jit
import traceback
from molseeq.funcs.pixseq_utils_compute import Worker
import concurrent.futures
import multiprocessing
import time
from functools import partial

def temporal_filtering(dat):

    try:

        filter_size = dat["filter_size"]
        filter_mode = dat["filter_mode"]
        h1,h2 = dat["height_range"]

        shared_mem = dat["shared_mem"]
        np_array = np.ndarray(dat["shape"], dtype=dat["dtype"], buffer=shared_mem.buf)
        filter_chunk = np_array[:,h1:h2,:]

        filter_chunk = image_temporal_filtering_jit(filter_chunk, filter_size, filter_mode)
        np_array[:,h1:h2,:] = filter_chunk

    except:
        print(traceback.format_exc())
        pass

@jit(nopython=True)
def pixel_temporal_filtering_jit(pixel_values, filter_size, filter_mode):
    n_frames = pixel_values.shape[0]
    filtered_values = np.empty_like(pixel_values)

    for k in range(n_frames):
        # Determine the start and end of the window
        start_frame = max(0, k - filter_size)
        end_frame = min(n_frames, k + filter_size + 1)

        # Special handling for the first frame
        if k < filter_size:
            if "mean" in filter_mode.lower():
                filtered_values[k] = np.mean(pixel_values[:end_frame])
            else:
                filtered_values[k] = np.median(pixel_values[:end_frame])
        else:
            if "mean" in filter_mode.lower():
                filtered_values[k] = np.mean(pixel_values[start_frame:end_frame])
            else:
                filtered_values[k] = np.median(pixel_values[start_frame:end_frame])

    return filtered_values

@jit(nopython=True)
def image_temporal_filtering_jit(image, filter_size, filter_mode):

    n_frames, height, width = image.shape

    for h_index in range(height):
        for w_index in range(width):

            pixel_values = image[:, h_index, w_index].copy()

            pixel_values = pixel_temporal_filtering_jit(pixel_values, filter_size, filter_mode)

            image[:, h_index, w_index] = pixel_values

    return image


class _utils_temporal_filtering:


    def update_filtering_channels(self):

        try:

            if self.dataset_dict != {}:

                filtering_datasets = self.gui.filtering_datasets.currentText()

                if filtering_datasets == "All Datasets":
                    dataset_names = list(self.dataset_dict.keys())
                    channel_names = list(self.dataset_dict[dataset_names[0]].keys())
                else:
                    dataset_names = [filtering_datasets]
                    channel_names = list(self.dataset_dict[dataset_names[0]].keys())

                for channel_index, channel_name in enumerate(channel_names):

                    if channel_name.lower() in ["dd","da","ad","aa"]:
                        channel_name = channel_name.upper()
                    else:
                        channel_name = channel_name.capitalize()

                    channel_names[channel_index] = channel_name

                if len(channel_names) > 0:
                    channel_names.insert(0, "All Channels")

                self.gui.filtering_channels.clear()
                self.gui.filtering_channels.addItems(channel_names)

        except:
            print(traceback.format_exc())
            pass

    def calculate_image_chunks(self, total_height, num_chunks):

        chunk_size = total_height // num_chunks
        height_ranges = []

        for i in range(num_chunks):
            start_height = i * chunk_size
            # Ensure the last chunk extends to the end of the image
            end_height = total_height if i == num_chunks - 1 else start_height + chunk_size
            height_ranges.append((start_height, end_height))

        return height_ranges

    def _populate_temport_compute_jobs(self):

        filter_size = int(self.gui.filtering_filter_size.currentText())
        filter_mode = self.gui.filtering_mode.currentText()

        compute_jobs = []

        try:
            for image_dict in self.shared_images:

                h = image_dict['shape'][1]

                height_ranges = self.calculate_image_chunks(h, 10)

                for height_range in height_ranges:

                    compute_job = {"filter_size": filter_size,
                                   "filter_mode": filter_mode,
                                   "shared_mem": image_dict['shared_mem'],
                                   "shape": image_dict['shape'],
                                   "dtype": image_dict['dtype'],
                                   "height_range": height_range,
                                   "stop_event": self.stop_event,
                                   }

                    compute_jobs.append(compute_job)

        except:
            print(traceback.format_exc())
            pass

        return compute_jobs

    def _pixseq_temporal_filtering(self, progress_callback = True):

        try:

            filtering_datasets = self.gui.filtering_datasets.currentText()
            filtering_channels = self.gui.filtering_channels.currentText()

            if filtering_datasets == "All Datasets":
                dataset_names = list(self.dataset_dict.keys())
            else:
                dataset_names = [filtering_datasets]

            if filtering_channels == "All Channels":
                channel_names = None
            else:
                channel_names = [filtering_channels.lower()]

            self.shared_images = self.create_shared_images(dataset_names, channel_names)

            compute_jobs = self._populate_temport_compute_jobs()

            self.pixseq_notification("Starting temporal filtering on {} images".format(len(self.shared_images)))

            start_time = time.time()

            if len(compute_jobs) == 0:
                pass
            elif len(compute_jobs) == 1:
                temporal_filtering(compute_jobs[0])
            else:
                cpu_count = int(multiprocessing.cpu_count() * 0.9)
                timeout_duration = 10  # Timeout in seconds

                with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                    # Submit all jobs and store the future objects
                    futures = {executor.submit(temporal_filtering, job): job for job in compute_jobs}

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
                            progress = int((iter / len(compute_jobs)) * 100)
                            progress_callback.emit(progress)  # Emit the signal



            self.restore_shared_images()

        except:
            self.update_ui()
            print(traceback.format_exc())
            pass

    def pixseq_temporal_filtering(self, viewer=None):

        try:
            self.pixseq_notification("Starting temporal filtering...")

            self.gui.filtering_start.setEnabled(False)

            self.update_ui(init=True)

            self.worker = Worker(self._pixseq_temporal_filtering)
            self.worker.signals.progress.connect(partial(self.pixseq_progress, progress_bar=self.gui.filtering_progressbar))
            self.worker.signals.finished.connect(self._pixseq_temporal_filtering_finished)
            self.worker.signals.error.connect(self.update_ui)
            self.threadpool.start(self.worker)

        except:
            self.update_ui()
            print(traceback.format_exc())
            pass

    def _pixseq_temporal_filtering_finished(self):

        try:
            self.image_layer.data = self.dataset_dict[self.active_dataset][self.active_channel]["data"]

            self.update_ui()

        except:
            print(traceback.format_exc())
            pass





    # @jit(nopython=True)
    # def pixseq_temporal_filtering(self, viewer=None):
    #
    #     dataset = self.active_dataset
    #     channel = self.active_channel
    #     filter_size = 5
    #
    #     print(f"Applying temporal filtering to {channel}...")
    #
    #     image = self.dataset_dict[dataset][channel]['data'].copy()
    #
    #     num_frames, height, width = image.shape
    #
    #     n_pixels = height * width * num_frames
    #
    #     iter = 0
    #
    #     for i in range(height):
    #         for j in range(width):
    #             # For each pixel, iterate over each frame
    #             for k in range(num_frames):
    #                 # Determine the start and end of the window
    #                 start_frame = max(0, k - filter_size)
    #                 end_frame = min(num_frames, k + filter_size + 1)
    #
    #                 # Extract the pixel values from the window and compute the median
    #                 pixel_values = image[start_frame:end_frame, i, j]
    #                 image[k, i, j] = np.median(pixel_values)
    #
    #                 iter += 1
    #
    #         print(f'Progress: {(iter/n_pixels)*100}', end='\r')
    #
    #     self.dataset_dict[dataset][channel]['data'] = image
    #     self.update_active_image()





