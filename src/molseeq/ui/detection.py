import traceback
from dataclasses import asdict
from functools import partial

import numpy as np
from molcore import ArraySource, LocalisationSettings
from numpy.lib.recfunctions import stack_arrays

from molseeq.model_analysis import render_localisations
from molseeq.model_processing import LocalisationJob, localise_channels
from molseeq.ui.worker import Worker


class DetectionMixin:

    def _picasso_wrapper_finished(self):

        try:

            image_channel = self.gui.picasso_channel.currentText()
            dataset_name = self.gui.picasso_dataset.currentText()

            if dataset_name == "All Datasets":
                dataset_name = self.active_dataset

            self.update_active_image(channel=image_channel.lower(), dataset=dataset_name)

            self.draw_bounding_boxes()
            self.draw_localisations()

            self.update_filter_criterion()
            self.update_criterion_ranges()

        except:
            print(traceback.format_exc())
        finally:
            self.update_ui()

    def molseeq_picasso(self, detect = False, fit = False):
        image_cleared = False
        try:
            if self.dataset_dict != {}:

                dataset_name = self.gui.picasso_dataset.currentText()
                min_net_gradient = self.gui.picasso_min_net_gradient.text()
                image_channel = self.gui.picasso_channel.currentText()
                frame_mode = self.gui.picasso_frame_mode.currentText()
                minimise_ram = self.gui.picasso_minimise_ram.isChecked()
                picasso_use_gpufit = self.gui.picasso_use_gpufit.isChecked()

                if self.gpufit_available and picasso_use_gpufit:
                    gpu_fit = True
                else:
                    gpu_fit = False

                if min_net_gradient.isdigit() and image_channel != "":

                    self.gui.picasso_progressbar.setValue(0)
                    self.gui.picasso_detect.setEnabled(False)
                    self.gui.picasso_fit.setEnabled(False)
                    self.gui.picasso_detectfit.setEnabled(False)

                    self.update_ui(init=True)

                    if minimise_ram == True and frame_mode.lower() != "active":
                        image_cleared = self._clear_detection_image()

                    if frame_mode.lower() == "active":
                        frame_index = self.viewer.dims.current_step[0]
                    else:
                        frame_index = None

                    if dataset_name == "All Datasets":
                        dataset_list = list(self.dataset_dict.keys())
                    else:
                        dataset_list = [dataset_name]

                    self._start_core_localisation(
                        dataset_list, image_channel.lower(), int(min_net_gradient),
                        frame_index, detect, fit, gpu_fit=gpu_fit and fit,
                    )

        except:
            print(traceback.format_exc())

            if image_cleared:
                self._picasso_wrapper_finished()
            else:
                self.update_ui()

    def _clear_detection_image(self):
        layer = getattr(self, "image_layer", None)
        if layer is None or layer not in self.viewer.layers:
            return False
        layer.data = np.zeros(layer.data.shape[-2:], dtype=layer.data.dtype)
        return True

    def _start_core_localisation(self, names, channel, threshold, frame, detect, fit, *, gpu_fit):
        mode = self.gui.picasso_detect_mode.currentText().lower()
        box_size = int(self.gui.picasso_box_size.currentText())
        settings = LocalisationSettings(min_net_gradient=threshold, box_size=box_size)
        jobs = []
        previous = {}
        for name in names:
            if channel not in self.session.images[name]:
                continue
            source = self.session.working_source(name, channel)
            entry, _, _ = self.get_loc_dict(name, channel, type=mode)
            previous[name] = entry.get("localisations")
            dataset = self.session.dataset(name).snapshot()
            for view, offset in self.session.channel_parts(name, channel):
                if not len(view) or (frame is not None and not offset <= frame < offset + len(view)):
                    continue
                seeds = None
                if not detect:
                    if previous[name] is None:
                        continue
                    seeds = previous[name]
                    if mode == "bounding boxes" and frame is not None:
                        seeds = seeds.copy()
                        seeds.frame = frame
                    seeds = seeds[(seeds.frame >= offset) & (seeds.frame < offset + len(view))]
                    if frame is not None:
                        seeds = seeds[seeds.frame == frame]
                working = (
                    ArraySource(source.array[offset:offset + len(view)])
                    if isinstance(source, ArraySource) else None
                )
                jobs.append((name, LocalisationJob(dataset, view, offset, frame, working, seeds)))
        if not jobs:
            self._picasso_wrapper_finished()
            return

        self.worker = Worker(
            localise_channels, jobs, settings, detect=detect, fit=fit, gpu_fit=gpu_fit,
            roi=self.generate_roi(),
            remove_overlapping=self.gui.picasso_remove_overlapping.isChecked(),
            cancel_event=self.stop_event,
        )
        self.worker.signals.progress.connect(self.gui.picasso_progressbar.setValue)
        self.worker.signals.result.connect(partial(
            self._publish_core_localisations, revision=self.session.revision,
            channel=channel, mode=mode, frame=frame, fitted=fit, box_size=box_size,
            previous=previous, sites=self.bounding_box_token(), creation_settings=asdict(settings),
        ))
        self.worker.signals.error.connect(lambda error: self.molseeq_notification(str(error[1])))
        self.worker.signals.finished.connect(self._picasso_wrapper_finished)
        self.threadpool.start(self.worker)

    def _publish_core_localisations(
        self, results, revision, channel, mode, frame, fitted, box_size, previous, sites, creation_settings
    ):
        if self.stop_event.is_set() or self.session.revision != revision:
            return
        if mode == "bounding boxes" and self.bounding_box_token() != sites:
            return
        entries = []
        for name, rows in results.items():
            if mode == "bounding boxes":
                if len(rows):
                    self.localisation_dict["bounding_boxes"] = {
                        "localisations": rows, "fitted": fitted, "box_size": box_size,
                        "origin": "picasso_fit" if fitted else "picasso_detect",
                        "source": {"dataset": name, "channel": channel, "frame": frame,
                                   "settings": creation_settings},
                    }
                continue
            old = previous[name]
            if frame is not None and old is not None:
                defaults = {
                    key: np.nan for chunk in (old, rows)
                    for key in chunk.dtype.names if chunk.dtype[key].kind == "f"
                }
                rows = stack_arrays(
                    (old[old.frame != frame], rows), defaults=defaults,
                    usemask=False, asrecarray=True, autoconvert=True,
                )
            entries.append((name, channel, {
                "localisations": rows, "fitted": fitted, "box_size": box_size,
            }))
        if entries:
            try:
                self.session.publish_localisations(entries)
            except ValueError as error:
                self.molseeq_notification(str(error))


    def generate_roi(self):

        if self.verbose:
            print("Generating ROI")

        border_width = self.gui.picasso_roi_border_width.text()
        window_cropping = self.gui.picasso_window_cropping .isChecked()

        roi = None

        try:

            generate_roi = False

            if window_cropping:
                layers_names = [layer.name for layer in self.viewer.layers if layer.name not in ["bounding_boxes", "localisations"]]

                crop = self.viewer.layers[layers_names[0]].corner_pixels[:, -2:]
                [[y1, x1], [y2, x2]] = crop

                generate_roi = True

            else:

                if type(border_width) == str:
                    border_width = int(border_width)
                    if border_width > 0:
                        generate_roi = True
                elif type(border_width) == int:
                    if border_width > 0:
                        generate_roi = True

            if generate_roi:

                dataset = self.gui.picasso_dataset.currentText()
                channel = self.gui.picasso_channel.currentText()

                if dataset == "All Datasets":
                    dataset = list(self.dataset_dict.keys())[0]

                image_shape = self.dataset_dict[dataset][channel.lower()]["data"].shape

                frame_shape = image_shape[1:]

                if window_cropping:

                    border_width = int(border_width)

                    if x1 < border_width:
                        x1 = border_width
                    if y1 < border_width:
                        y1 = border_width
                    if x2 > frame_shape[1] - border_width:
                        x2 = frame_shape[1] - border_width
                    if y2 > frame_shape[0] - border_width:
                        y2 = frame_shape[0] - border_width

                    roi = [[y1, x1], [y2, x2]]

                else:

                    roi = [[int(border_width), int(border_width)],
                           [int(frame_shape[0] - border_width), int(frame_shape[1] - border_width)]]

        except:
            print(traceback.format_exc())
            pass

        return roi

    def picasso_render_finished(self):

        self.update_filter_criterion()
        self.update_criterion_ranges()

        self.update_ui(init=False)



    def draw_picasso_render(self, data):

        try:
            image, pixel_size, oversampling = data

            scale = [pixel_size / oversampling, pixel_size / oversampling]

            layer_names = [layer.name for layer in self.viewer.layers]

            if "SMLM Render" not in layer_names:
                self.viewer.add_image(image, name="SMLM Render", colormap="viridis", scale=scale, )
            else:
                self.viewer.layers["SMLM Render"].data = image
                self.viewer.layers["SMLM Render"].scale = scale

        except:
            print(traceback.format_exc())

    def _publish_picasso_render(self, data, *, revision):
        if self.session.revision == revision and not self.stop_event.is_set():
            self.draw_picasso_render(data)

    def initialise_picasso_render(self):

        try:

            dataset_name = self.gui.picasso_render_dataset.currentText()
            image_channel = self.gui.picasso_render_channel.currentText()
            blur_method = self.gui.picasso_render_blur_method.currentText()
            min_blur_width = float(self.gui.picasso_render_min_blur.text())

            if (image_channel.lower() in self.localisation_dict["localisations"][dataset_name].keys()):
                localisation_dict = self.localisation_dict["localisations"][dataset_name][image_channel.lower()].copy()

                if localisation_dict["fitted"] == True:
                    locs = localisation_dict["localisations"]

                    image_shape = self.session.working_source(dataset_name, image_channel.lower()).shape
                    revision = self.session.revision

                    if blur_method == "One-Pixel-Blur":
                        blur_method = "smooth"
                    elif blur_method == "Global Localisation Precision":
                        blur_method = "convolve"
                    elif (blur_method == "Individual Localisation Precision, iso"):
                        blur_method = "gaussian_iso"
                    elif blur_method == "Individual Localisation Precision":
                        blur_method = "gaussian"
                    else:
                        blur_method = None

                    self.update_ui(init=True)

                    worker = Worker(
                        render_localisations, locs, image_shape,
                        blur_method=blur_method, min_blur_width=min_blur_width,
                        cancel_event=self.stop_event,
                    )
                    worker.signals.result.connect(partial(
                        self._publish_picasso_render, revision=revision
                    ))
                    worker.signals.error.connect(
                        lambda error: self.molseeq_notification(str(error[1]))
                        if not self.stop_event.is_set() else None
                    )
                    worker.signals.finished.connect(self.picasso_render_finished)
                    self.threadpool.start(worker)

        except:
            print(traceback.format_exc())
            self.update_ui()
