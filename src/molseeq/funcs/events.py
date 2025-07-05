import os.path
import traceback
import numpy as np
from functools import partial, wraps
from qtpy.QtWidgets import (QSlider, QLabel)
import time
from napari.utils.notifications import show_info

class _events_utils:

    def molseeq_notification(self, message):
        show_info(message)

    def update_ui(self, error=None, init = False):

        try:

            if self.verbose:
                print(f"Updating UI, init = {init}")

            controls = ["molseeq_import",
                        "picasso_detect", "picasso_fit", "picasso_detectfit",
                        "molseeq_compute_tform", "molseeq_apply_tform",
                        "picasso_undrift","molseeq_align_datasets",
                        "filtering_start",
                        "compute_traces",
                        "molseeq_export_data","molseeq_export_traces",
                        "molseeq_update_dataset_name",
                        "molseeq_colocalize",
                        "cluster_localisations",
                        "filter_localisations",
                        "link_localisations",
                        ]

            progressbars = ["molseeq_import_progressbar",
                            "picasso_progressbar",
                            "tform_apply_progressbar",
                            "undrift_progressbar",
                            "align_progressbar",
                            "filtering_progressbar",
                            "compute_traces_progressbar",
                            "plot_compute_progress",
                            "export_progressbar",
                            ]

            for progressbar in progressbars:
                if hasattr(self.gui, progressbar):
                    getattr(self.gui, progressbar).setValue(0)

            if init is True:

                for control in controls:
                    getattr(self.gui, control).setEnabled(False)

                self.stop_event.clear()
                self.multiprocessing_active = True

            else:

                for control in controls:
                    getattr(self.gui, control).setEnabled(True)

                self.multiprocessing_active = False

                self.stop_event.clear()
                self.multiprocessing_active = False

            if error is not None:
                print(error)

        except:
            print(traceback.format_exc())
            pass


    def stop_worker(self, viewer=None):

        if self.stop_event is not None:
            self.stop_event.set()

        while self.multiprocessing_active is True:
            time.sleep(0.1)

        if self.stop_event is not None:
            self.stop_event.clear()

        if self.worker is not None:
            self.worker.stop()

        self.update_ui()


    def increment_active_dataset(self, viewer=None, key=None):
        try:
            if self.dataset_dict != {}:
                dataset_list = list(self.dataset_dict.keys())
                current_dataset = self.active_dataset
                current_dataset_index = dataset_list.index(current_dataset)

                if key == 'Up':
                    new_dataset_index = current_dataset_index + 1
                elif key == 'Down':
                    new_dataset_index = current_dataset_index - 1

                if new_dataset_index < 0:
                    new_dataset_index = len(dataset_list) - 1
                elif new_dataset_index > len(dataset_list) - 1:
                    new_dataset_index = 0

                self.gui.molseeq_dataset_selector.setCurrentIndex(new_dataset_index)

        except:
            print(traceback.format_exc())
            pass

    def named_partial(self, func, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)

        # Use wraps to copy metadata from the original function to the partial function
        @wraps(func)
        def wrapper(*args, **kwargs):
            return partial_func(*args, **kwargs)

        return wrapper

    def update_overlay_text(self):

        try:

            if self.dataset_dict  != {}:

                dataset_name = self.gui.molseeq_dataset_selector.currentText()
                channel_name = self.active_channel

                if dataset_name in self.dataset_dict.keys():
                    if channel_name in self.dataset_dict[dataset_name].keys():

                        channel_dict = self.dataset_dict[dataset_name][channel_name].copy()

                        gap_label = channel_dict["gap_label"]
                        sequence_label = channel_dict["sequence_label"]

                        path = channel_dict["path"]
                        file_name = os.path.basename(path)

                        if channel_name in ["da", "dd", "aa", "ad"]:
                            channel_name = channel_name.upper()
                        else:
                            channel_name = channel_name.capitalize()

                        overlay_string = ""
                        overlay_string += f"File: {file_name}\n"
                        overlay_string += f"Dataset: {dataset_name}\n"
                        overlay_string += f"Channel: {channel_name}\n"
                        overlay_string += f"Gap label: {gap_label}\n"
                        overlay_string += f"Sequence label: {sequence_label}\n"

                        if overlay_string != "":
                            self.viewer.text_overlay.visible = True
                            self.viewer.text_overlay.position = "top_left"
                            self.viewer.text_overlay.text = overlay_string.lstrip("\n")
                            self.viewer.text_overlay.color = "red"
                            self.viewer.text_overlay.font_size = 9
                        else:
                            self.viewer.text_overlay.visible = False

        except:
            print(traceback.format_exc())


    def select_channel_da(self, event=None):
        self.update_active_image(channel="da")

    def select_channel_dd(self, event=None):
        self.update_active_image(channel="dd")

    def select_channel_aa(self, event=None):
        self.update_active_image(channel="aa")

    def select_channel_ad(self, event=None):
        self.update_active_image(channel="ad")

    def select_channel_donor(self, event=None):
        self.update_active_image(channel="donor")

    def select_channel_acceptor(self, event=None):
        self.update_active_image(channel="acceptor")


    def image_layer_auto_contrast(self, image, dataset, channel):

        contrast_limits = None

        try:
            autocontrast = True

            if dataset in self.contrast_dict.keys():
                if channel in self.contrast_dict[dataset].keys():

                    autocontrast = False

                    contrast_limits = self.contrast_dict[dataset][channel]["contrast_limits"]
                    gamma = self.contrast_dict[dataset][channel]["gamma"]

                    if hasattr(self, "image_layer"):
                        self.image_layer.gamma = gamma
                        self.image_layer.contrast_limits = contrast_limits

            if autocontrast is True:

                full_range = [np.min(image), np.max(image)]

                if max(full_range) > min(full_range):
                    contrast_limits = np.percentile(image[:10].copy(), [0.1, 99.99])

                    gamma = 1.0
                    if contrast_limits[1] > contrast_limits[0]:
                        gamma = np.log(0.5) / np.log((contrast_limits[1] - contrast_limits[0]) / (full_range[1] - full_range[0]))

                    if hasattr(self, "image_layer"):
                        self.image_layer.gamma = gamma
                        self.image_layer.contrast_limits = contrast_limits

        except:
            print(traceback.format_exc())

        return contrast_limits

    def update_contrast_dict(self):

        try:
            dataset = self.active_dataset
            channel = self.active_channel

            if dataset not in self.contrast_dict.keys():
                self.contrast_dict[dataset] = {}
            if channel not in self.contrast_dict[dataset].keys():
                self.contrast_dict[dataset][channel] = {}

            layer_name = [layer.name for layer in self.viewer.layers if dataset in layer.name]

            if len(layer_name) > 0:

                image_layer = self.viewer.layers[layer_name[0]]
                contrast_limits = image_layer.contrast_limits
                gamma = image_layer.gamma

                self.contrast_dict[dataset][channel] = {"contrast_limits": contrast_limits,
                                                        "gamma": gamma}

        except:
            print(traceback.format_exc())

    def update_active_image(self, channel=None, dataset=None, event=None):

        try:

            if self.verbose:
                print(f"Updating active image to channel {channel} and dataset {dataset}")

            if dataset == None or dataset not in self.dataset_dict.keys():
                dataset_name = self.gui.molseeq_dataset_selector.currentText()
            else:
                dataset_name = dataset

            if dataset_name in self.dataset_dict.keys():

                self.update_contrast_dict()

                channel_names = [channel for channel in self.dataset_dict[dataset_name].keys()]

                if channel not in channel_names:
                    if self.active_channel in channel_names:
                        channel = self.active_channel
                    else:
                        channel = channel_names[0]

                self.active_dataset = dataset_name
                self.active_channel = channel

                if "data" in self.dataset_dict[dataset_name][channel].keys():

                    image = self.dataset_dict[dataset_name][channel]["data"]

                    if channel in ["da", "dd", "aa", "ad"]:
                        channel_name = channel.upper()
                    else:
                        channel_name = channel.capitalize()

                    layer_name = f"{dataset_name}: [{channel_name}]"

                    if hasattr(self, "image_layer") == False:

                        self.image_layer = self.viewer.add_image(image,
                            name=layer_name,
                            colormap="gray",
                            blending="additive",
                            visible=True)

                        self.image_layer.mouse_drag_callbacks.append(self._mouse_event)

                    else:
                        self.image_layer.data = image
                        self.image_layer.name = layer_name
                        self.image_layer.refresh()

                    self.image_layer_auto_contrast(image, dataset_name, channel)

                    dataset_names = self.dataset_dict.keys()
                    active_dataset_index = list(dataset_names).index(dataset_name)

                    dataset_selector = self.gui.molseeq_dataset_selector

                    dataset_selector.blockSignals(True)
                    dataset_selector.clear()
                    dataset_selector.addItems(dataset_names)
                    dataset_selector.setCurrentIndex(active_dataset_index)
                    dataset_selector.blockSignals(False)

            else:
                if hasattr(self, "image_layer"):
                    self.viewer.layers.remove(self.image_layer)

                self.active_dataset = None
                self.active_channel = None

            self.draw_localisations(update_vis=True)
            self.update_overlay_text()

        except:
            print(traceback.format_exc())
            pass

    def update_channel_selector(self, dataset_selector,
            channel_selector, event=None, channel_type = "all", efficiency=False, block_signals=False):

        try:

            if self.verbose:
                print(f"Updating channel selector for dataset {dataset_selector} and channel {channel_selector}")

            if hasattr(self.gui, channel_selector) and hasattr(self.gui, dataset_selector):

                channel_selector = getattr(self.gui, channel_selector)
                dataset_selector = getattr(self.gui, dataset_selector)

                dataset_name = dataset_selector.currentText()

                if block_signals == True:
                    channel_selector.blockSignals(True)

                channel_selector_list = []

                if dataset_name in self.dataset_dict.keys():

                    channel_names = [channel.lower() for channel in self.dataset_dict[dataset_name].keys()]

                    if channel_type.lower() == "donor":
                        channel_names = [channel for channel in channel_names if channel in ["dd","ad", "donor"]]
                    elif channel_type.lower() == "acceptor":
                        channel_names = [channel for channel in channel_names if channel in ["da","aa", "acceptor"]]

                    for channel in channel_names:

                        if "efficiency" not in channel.lower():

                            if channel in ["da", "dd", "aa", "ad"]:
                                channel_selector_list.append(channel.upper())
                            elif channel in ["donor", "acceptor","data"]:
                                channel_selector_list.append(channel.capitalize())

                            if efficiency == True:
                                if set(["donor", "acceptor"]).issubset(set(channel_names)):
                                    channel_selector_list.append("FRET Efficiency")
                                if set(["dd", "da"]).issubset(set(channel_names)):
                                    channel_selector_list.append("ALEX Efficiency")

                elif dataset_name == "All Datasets":

                    channel_names = []

                    for dataset_name in self.dataset_dict.keys():
                        dataset_channels = [channel.lower() for channel in self.dataset_dict[dataset_name].keys()]
                        channel_names.append(dataset_channels)

                    channel_names = list(set.intersection(*map(set, channel_names)))

                    for channel in channel_names:

                        if "efficiency" not in channel.lower():

                            if channel in ["da", "dd", "aa", "ad"]:
                                channel_selector_list.append(channel.upper())
                            elif channel in ["donor", "acceptor", "data"]:
                                channel_selector_list.append(channel.capitalize())

                            if efficiency == True:
                                if set(["donor", "acceptor"]).issubset(set(channel_names)):
                                    channel_selector_list.append("FRET Efficiency")
                                if set(["dd", "da"]).issubset(set(channel_names)):
                                    channel_selector_list.append("ALEX Efficiency")

                if channel_selector_list != []:

                    channel_selector.clear()
                    channel_selector_list = list(set(channel_selector_list))
                    channel_selector.addItems(channel_selector_list)

                channel_selector.blockSignals(False)

        except:
            print(traceback.format_exc())
            pass

    def populate_channel_selectors(self):

        try:

            if self.verbose:
                print("Populating channel selectors")

            self.update_channel_selector(dataset_selector="picasso_dataset", channel_selector="picasso_channel")
            self.update_channel_selector(dataset_selector="import_picasso_dataset", channel_selector="import_picasso_channel", efficiency=False)
            self.update_channel_selector(dataset_selector="aim_dataset_selector", channel_selector="aim_channel_selector")
            self.update_channel_selector(dataset_selector="cluster_dataset", channel_selector="cluster_channel")
            self.update_channel_selector(dataset_selector="tform_compute_dataset", channel_selector="tform_compute_ref_channel", channel_type="donor")
            self.update_channel_selector(dataset_selector="tform_compute_dataset", channel_selector="tform_compute_target_channel", channel_type="acceptor")
            self.update_channel_selector(dataset_selector="colo_dataset", channel_selector="colo_channel1")
            self.update_channel_selector(dataset_selector="colo_dataset", channel_selector="colo_channel2")
            self.update_channel_selector(dataset_selector="simple_plot_dataset", channel_selector="simple_plot_channel", efficiency=True)
            self.update_channel_selector(dataset_selector="picasso_render_dataset", channel_selector="picasso_render_channel")
            self.update_channel_selector(dataset_selector="picasso_filter_dataset", channel_selector="picasso_filter_channel")
            self.update_channel_selector(dataset_selector="tracking_dataset", channel_selector="tracking_channel")

        except:
            print(traceback.format_exc())
            pass



    def update_channel_select_buttons(self):

        try:
            datast_name = self.gui.molseeq_dataset_selector.currentText()

            if datast_name in self.dataset_dict.keys():

                if self.verbose:
                    print(f"Updating channel select buttons for dataset {datast_name}")

                import_modes = [self.dataset_dict[datast_name][channel]["import_mode"] for channel in self.dataset_dict[datast_name].keys()]
                fret_modes = [self.dataset_dict[datast_name][channel]["FRET"] for channel in self.dataset_dict[datast_name].keys()]
                channel_refs = [self.dataset_dict[datast_name][channel]["channel_ref"] for channel in self.dataset_dict[datast_name].keys()]

                channel_refs = list(set(channel_refs))
                fret_mode = list(set(fret_modes))[0]
                import_mode = list(set(import_modes))[0]

                self.gui.molseeq_show_dd.clicked.connect(lambda: None)
                self.gui.molseeq_show_da.clicked.connect(lambda: None)
                self.gui.molseeq_show_aa.clicked.connect(lambda: None)
                self.gui.molseeq_show_ad.clicked.connect(lambda: None)

                if import_mode.lower() == "fret":

                    self.gui.molseeq_show_dd.setVisible(True)
                    self.gui.molseeq_show_da.setVisible(True)
                    self.gui.molseeq_show_aa.setVisible(False)
                    self.gui.molseeq_show_ad.setVisible(False)

                    if "dd" in channel_refs:
                        self.gui.molseeq_show_dd.setEnabled(True)
                        self.gui.molseeq_show_dd.setText("Donor [F1]")
                        self.viewer.bind_key("F1", func=self.select_channel_donor, overwrite=True)
                        self.gui.molseeq_show_dd.clicked.connect(partial(self.update_active_image, channel="donor"))
                    else:
                        self.gui.molseeq_show_dd.setEnabled(False)
                        self.gui.molseeq_show_dd.setText("")

                    if "da" in channel_refs:
                        self.gui.molseeq_show_da.setEnabled(True)
                        self.gui.molseeq_show_da.setText("Acceptor [F2]")
                        self.viewer.bind_key("F2", func=self.select_channel_acceptor, overwrite=True)
                        self.gui.molseeq_show_da.clicked.connect(partial(self.update_active_image, channel="acceptor"))
                    else:
                        self.gui.molseeq_show_da.setEnabled(False)
                        self.gui.molseeq_show_da.setText("")

                elif import_mode.lower() == "single channel":

                    self.gui.molseeq_show_dd.setVisible(False)
                    self.gui.molseeq_show_da.setVisible(False)
                    self.gui.molseeq_show_aa.setVisible(False)
                    self.gui.molseeq_show_ad.setVisible(False)

                else:

                    self.gui.molseeq_show_dd.setVisible(True)
                    self.gui.molseeq_show_da.setVisible(True)
                    self.gui.molseeq_show_aa.setVisible(True)
                    self.gui.molseeq_show_ad.setVisible(True)

                    if "dd" in channel_refs:
                        self.gui.molseeq_show_dd.setText("DD [F1]")
                        self.gui.molseeq_show_dd.setEnabled(True)
                        self.viewer.bind_key("F1", func=self.select_channel_dd, overwrite=True)
                        self.gui.molseeq_show_dd.clicked.connect(partial(self.update_active_image, channel="dd"))

                    else:
                        self.gui.molseeq_show_dd.setText("")
                        self.gui.molseeq_show_dd.setEnabled(False)

                    if "da" in channel_refs:
                        self.gui.molseeq_show_da.setText("DA [F2]")
                        self.gui.molseeq_show_da.setEnabled(True)
                        self.viewer.bind_key("F2", func=self.select_channel_da, overwrite=True)
                        self.gui.molseeq_show_da.clicked.connect(partial(self.update_active_image, channel="da"))
                    else:
                        self.gui.molseeq_show_da.setText("")
                        self.gui.molseeq_show_da.setEnabled(False)

                    if "ad" in channel_refs:
                        self.gui.molseeq_show_ad.setText("AD [F3]")
                        self.gui.molseeq_show_ad.setEnabled(True)
                        self.viewer.bind_key("F3", func=self.select_channel_ad, overwrite=True)
                        self.gui.molseeq_show_ad.clicked.connect(partial(self.update_active_image, channel="ad"))
                    else:
                        self.gui.molseeq_show_ad.setText("")
                        self.gui.molseeq_show_ad.setEnabled(False)

                    if "aa" in channel_refs:
                        self.gui.molseeq_show_aa.setText("AA [F4]")
                        self.gui.molseeq_show_aa.setEnabled(True)
                        self.viewer.bind_key("F4", func=self.select_channel_aa, overwrite=True)
                        self.gui.molseeq_show_aa.clicked.connect(partial(self.update_active_image, channel="aa"))
                    else:
                        self.gui.molseeq_show_aa.setText("")
                        self.gui.molseeq_show_aa.setEnabled(False)

        except:
            print(traceback.format_exc())
            pass

    def update_import_append_options(self):

        try:

            if self.gui.molseeq_append.isChecked():
                self.gui.molseeq_append_dataset.setEnabled(True)
                self.gui.molseeq_append_dataset_label.setEnabled(True)
                self.gui.molseeq_append_dataset.setVisible(True)
                self.gui.molseeq_append_dataset_label.setVisible(True)
            else:
                self.gui.molseeq_append_dataset.setEnabled(False)
                self.gui.molseeq_append_dataset_label.setEnabled(False)
                self.gui.molseeq_append_dataset.setVisible(False)
                self.gui.molseeq_append_dataset_label.setVisible(False)

        except:
            print(traceback.format_exc())
            pass




    def update_import_options(self):

        if self.verbose:
            print("Updating import options")

        def update_channel_layout(self, show = True):
            if show:
                self.gui.molseeq_channel_layout.setEnabled(True)
                self.gui.molseeq_channel_layout.clear()
                self.gui.molseeq_channel_layout.addItems(["Donor-Acceptor", "Acceptor-Donor"])
                self.gui.molseeq_channel_layout.setHidden(False)
                self.gui.molseeq_channel_layout_label.setHidden(False)
            else:
                self.gui.molseeq_channel_layout.setEnabled(False)
                self.gui.molseeq_channel_layout.clear()
                self.gui.molseeq_channel_layout.setHidden(True)
                self.gui.molseeq_channel_layout_label.setHidden(True)

        def update_alex_first_frame(self, show = True):
            if show:
                self.gui.molseeq_alex_first_frame.setEnabled(True)
                self.gui.molseeq_alex_first_frame.clear()
                self.gui.molseeq_alex_first_frame.addItems(["Donor", "Acceptor"])
                self.gui.molseeq_alex_first_frame.setHidden(False)
                self.gui.molseeq_alex_first_frame_label.setHidden(False)
            else:
                self.gui.molseeq_alex_first_frame.setEnabled(False)
                self.gui.molseeq_alex_first_frame.clear()
                self.gui.molseeq_alex_first_frame.setHidden(True)
                self.gui.molseeq_alex_first_frame_label.setHidden(True)

        import_mode = self.gui.molseeq_import_mode.currentText()

        if import_mode in ["Donor", "Acceptor"]:
            update_channel_layout(self, show = False)
            update_alex_first_frame(self, show = False)

        elif import_mode == "FRET":
            update_channel_layout(self, show = True)
            update_alex_first_frame(self, show = False)

        elif import_mode == "ALEX":
            update_channel_layout(self, show = True)
            update_alex_first_frame(self, show = True)

        elif import_mode in ["DA", "DD", "AA", "AD"]:
            update_channel_layout(self, show = False)
            update_alex_first_frame(self, show = False)

        elif import_mode == "Single Channel":
            update_channel_layout(self, show = False)
            update_alex_first_frame(self, show = False)

    def molseeq_progress(self, progress, progress_bar):

        progress_bar.setValue(progress)

        if progress == 100:
            progress_bar.setValue(0)
            progress_bar.setHidden(True)
            progress_bar.setEnabled(False)
        else:
            progress_bar.setHidden(False)
            progress_bar.setEnabled(True)

    def _mouse_event(self, viewer, event):

        if self.verbose:
            print("Mouse event")

        try:

            event_pos = self.image_layer.world_to_data(event.position)
            image_shape = self.image_layer.data.shape
            modifiers = event.modifiers

            if "Shift" in modifiers or "Control" in modifiers:

                if "Shift" in modifiers:
                    mode = "localisations"
                elif "Control" in modifiers:
                    mode = "bounding_box"

                [y,x] = [event_pos[-2], event_pos[-1]]

                if (x >= 0) & (x < image_shape[-1]) & (y >= 0) & (y < image_shape[-2]):

                    self.add_manual_localisation(position=[x,y], mode=mode)

            if "Alt" in modifiers:

                [y, x] = [event_pos[-2], event_pos[-1]]
                self.add_lsp_localisation(position=[y,x])




        except:
            print(traceback.format_exc())


    def update_nucleotide(self):

        if self.verbose:
            print("Updating nucleotide")

        try:

            dataset_name = self.gui.update_labels_dataset.currentText()
            gap_label = self.gui.gap_label.currentText()
            sequence_label = self.gui.sequence_label.currentText()

            if dataset_name in self.dataset_dict.keys():
                for channel_dict in self.dataset_dict[dataset_name].values():
                    channel_dict["gap_label"] = gap_label
                    channel_dict["sequence_label"] = sequence_label

            if hasattr(self, "traces_dict"):
                if dataset_name in self.traces_dict.keys():
                    for channel_name, channel_dict in self.traces_dict[dataset_name].items():
                        for trace_index, trace_dict in channel_dict.items():
                            if "gap_label" in trace_dict.keys():
                                trace_dict["gap_label"] = gap_label
                                trace_dict["sequence_label"] = sequence_label

            self.update_overlay_text()

        except:
            print(traceback.format_exc())

    def delete_dataset(self):

        try:

            dataset_name = self.gui.delete_dataset_name.currentText()

            if dataset_name in self.dataset_dict.keys():

                if self.verbose:
                    print("Deleting dataset {dataset_name}")

                self.dataset_dict.pop(dataset_name)
                self.localisation_dict["localisations"].pop(dataset_name)

                if hasattr(self, "traces_dict"):
                    if dataset_name in self.traces_dict.keys():
                        self.traces_dict.pop(dataset_name)

                self.populate_dataset_combos()
                self.update_channel_select_buttons()
                self.update_active_image()

                self.populate_plot_combos()
                self.populate_export_combos()
                self.initialize_plot()

        except:
            print(traceback.format_exc())
            pass


    def update_dataset_name(self):

        try:

            old_name = self.gui.molseeq_old_dataset_name.currentText()
            new_name = self.gui.molseeq_new_dataset_name.text()

            if old_name != "":

                if new_name == "":
                    raise ValueError("New dataset name cannot be blank")
                elif new_name in self.dataset_dict.keys():
                    raise ValueError("New dataset name must be unique")
                else:

                    if self.verbose:
                        print("Updating dataset name from {old_name} to {new_name}")

                    dataset_data = self.dataset_dict.pop(old_name)
                    self.dataset_dict[new_name] = dataset_data

                    localisation_data = self.localisation_dict["localisations"].pop(old_name)
                    self.localisation_dict["localisations"][new_name] = localisation_data

                    if hasattr(self, "traces_dict"):
                        if old_name in self.traces_dict.keys():
                            print("Updating traces dict")
                            traces_data = self.traces_dict.pop(old_name)
                            self.traces_dict[new_name] = traces_data

                self.populate_dataset_combos()
                self.update_channel_select_buttons()
                self.update_active_image()

                self.populate_plot_combos()
                self.populate_export_combos()
                self.initialize_plot()

        except:
            print(traceback.format_exc())

    def update_slider_label(self, slider_name):

        label_name = slider_name + "_label"

        self.slider = self.findChild(QSlider, slider_name)
        self.gui.label = self.findChild(QLabel, label_name)

        slider_value = self.slider.value()
        self.gui.label.setText(str(slider_value))

    def update_picasso_options(self):

        if self.gui.picasso_detect_mode.currentText() == "Localisations":
            self.gui.picasso_frame_mode.clear()
            self.gui.picasso_frame_mode.addItems(["Active", "All"])
        else:
            self.gui.picasso_frame_mode.clear()
            self.gui.picasso_frame_mode.addItem("Active")