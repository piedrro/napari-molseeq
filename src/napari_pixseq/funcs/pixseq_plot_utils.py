from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import QSlider
from PyQt5.QtWidgets import QCheckBox
import numpy as np
import pyqtgraph as pg
import traceback
import copy
from qtpy.QtWidgets import QCheckBox
import re
from scipy.ndimage import gaussian_filter1d


class _plot_utils:

    def update_plot_combos(self, combo=""):

        if combo == "plot_data":
            self.update_plot_channel_combo()
            self.update_plot_metrics_combos()
        elif combo == "plot_channel":
            self.update_plot_metrics_combos()

    def populate_plot_combos(self):
        self.populate_plot_data_combo()
        self.update_plot_channel_combo()
        self.update_plot_metrics_combos()

    def populate_plot_data_combo(self):

        try:

            if hasattr(self, "traces_dict"):

                if self.traces_dict != {}:

                    self.updating_plot_combos = True

                    self.plot_data.blockSignals(True)
                    self.plot_data.clear()

                    dataset_names = list(self.traces_dict.keys())

                    if len(dataset_names) > 0:
                        dataset_names.insert(0, "All Datasets")

                    self.plot_data.addItems(dataset_names)
                    self.plot_data.blockSignals(False)

                    self.updating_plot_combos = False

        except:
            print(traceback.format_exc())

    def update_plot_channel_combo(self):

        try:

            if hasattr(self, "traces_dict"):

                if self.traces_dict != {}:

                    dataset_name = self.plot_data.currentText()

                    if dataset_name != "":

                        self.updating_plot_combos = True

                        self.plot_channel.blockSignals(True)
                        self.plot_channel.clear()

                        if dataset_name == "All Datasets":
                            channel_names = []
                            for dataset in self.traces_dict.keys():
                                channel_names.extend(list(self.traces_dict[dataset].keys()))
                            channel_names = list(set(channel_names))
                        else:
                            channel_names = list(self.traces_dict[dataset_name].keys())

                        plot_channel_list = ["All Channels"]

                        if set(["da", "dd", "aa", "ad"]).issubset(channel_names):
                            plot_channel_list.insert(1, "ALEX Data")
                            plot_channel_list.insert(2, "ALEX Efficiency")
                            plot_channel_list.insert(3, "ALEX Data + Efficiency")
                        if set(["donor", "acceptor"]).issubset(channel_names):
                            plot_channel_list.insert(1, "FRET Data")
                            plot_channel_list.insert(2, "FRET Efficiency")
                            plot_channel_list.insert(3, "FRET Data + Efficiency")

                        for channel_index, channel_name in enumerate(channel_names):
                            if channel_name.lower() in ["dd","da","ad","aa"]:
                                plot_channel_list.append(channel_name.upper())
                            elif channel_name.lower() in ["donor","acceptor"]:
                                plot_channel_list.append(channel_name.capitalize())
                            else:
                                pass

                        self.update_qcombo_items(self.plot_channel, plot_channel_list)

                        self.plot_channel.blockSignals(False)
                        self.updating_plot_combos = False

        except:
            print(traceback.format_exc())

    def update_plot_metrics_combos(self):

        try:

            if hasattr(self, "traces_dict"):

                if self.traces_dict != {}:

                    channel_names = []
                    dataset_names = list(self.traces_dict.keys())
                    for dataset in dataset_names:
                        channel_names.extend(list(self.traces_dict[dataset].keys()))

                    channel = channel_names[0]
                    dataset = dataset_names[0]

                    if dataset != "" and channel != "":

                        channel_dict = self.traces_dict[dataset][channel]
                        trace_indeces = list(channel_dict.keys())

                        n_traces = len(channel_dict)

                        if n_traces > 0:

                            metric_names = channel_dict[trace_indeces[0]].keys()

                            plot_metric_items = []
                            background_metric_items = []

                            for metric in metric_names:
                                if "_local_bg" in metric:
                                    background_metric_items.append("Local Background")
                                if "_masked_local_bg" in metric:
                                    background_metric_items.append("Masked Local Background")
                                if "_global_bg" in metric:
                                    background_metric_items.append("Global Background")
                                if "_masked_global_bg" in metric:
                                    background_metric_items.append("Masked Global Background")
                                if metric in self.metric_dict.keys():
                                    plot_metric_items.append(self.metric_dict[metric])

                            plot_metric_items = list(set(plot_metric_items))
                            background_metric_items = list(set(background_metric_items))

                            background_metric_items.insert(0, "None")

                            self.updating_plot_combos = True
                            self.update_qcombo_items(self.plot_metric, plot_metric_items)
                            self.update_qcombo_items(self.plot_background_mode, background_metric_items)
                            self.update_qcombo_items(self.gui.traces_export_metric, plot_metric_items)
                            self.update_qcombo_items(self.gui.traces_export_background, background_metric_items)
                            self.updating_plot_combos = False

        except:
            print(traceback.format_exc())


    def update_qcombo_items(self, qcombo, items):

        try:

            qcombo.blockSignals(True)

            current_items = [qcombo.itemText(i) for i in range(qcombo.count())]

            if current_items != items:
                qcombo.clear()
                qcombo.addItems(items)

            qcombo.blockSignals(False)

        except:
            print(traceback.format_exc())



    def get_dict_key(self, dict, target_value):

        dict_key = None

        if target_value not in ["None", None]:

            for key, value in dict.items():
                if value == target_value:
                    dict_key = key
                    break

        return dict_key

    def compute_fret_efficiency(self,  dataset_name, metric_key, background_metric_key,
            progress_callback=None, gamma_correction=1, clip_data=False, efficiency_offset=False):

        try:

            dataset_dict = self.traces_dict[dataset_name].copy()
            trace_indices = list(dataset_dict["donor"].keys())
            n_traces = len(trace_indices)

            dataset_dict["fret_efficiency"] = {}
            for trace_index in trace_indices:
                if trace_index not in dataset_dict["fret_efficiency"]:
                    dataset_dict["fret_efficiency"][trace_index] = {metric_key: []}

            for trace_index in trace_indices:

                donor = copy.deepcopy(dataset_dict["donor"][trace_index][metric_key])
                acceptor = copy.deepcopy(dataset_dict["acceptor"][trace_index][metric_key])

                if background_metric_key != None:

                    donor_bg = copy.deepcopy(dataset_dict["donor"][trace_index][background_metric_key])
                    acceptor_bg = copy.deepcopy(dataset_dict["acceptor"][trace_index][background_metric_key])

                    donor_bg = gaussian_filter1d(donor_bg, 1)
                    acceptor_bg = gaussian_filter1d(acceptor_bg, 1)

                    donor = donor - donor_bg
                    acceptor = acceptor - acceptor_bg

                if efficiency_offset:
                    global_min = np.min([np.min(donor), np.min(acceptor)])
                    global_min = abs(global_min)
                    donor = donor + global_min
                    acceptor = acceptor + global_min

                efficiency = acceptor / ((gamma_correction * donor) + acceptor)

                if clip_data:
                    efficiency = np.clip(efficiency, 0, 1)

                dataset_dict["fret_efficiency"][trace_index][metric_key] = efficiency

                if progress_callback is not None:
                    progress = int(100 * trace_index / n_traces)
                    progress_callback.emit(progress)

            self.traces_dict[dataset_name] = dataset_dict

        except:
            print(traceback.format_exc())
            pass

    def compute_alex_efficiency(self, dataset_name, metric_key, background_metric_key,
            progress_callback=None, gamma_correction=1, clip_data=False, efficiency_offset=False):

        try:

            dataset_dict = self.traces_dict[dataset_name].copy()
            n_traces = len(dataset_dict["dd"])

            dataset_dict["alex_efficiency"] = {}
            for trace_index in range(n_traces):
                if trace_index not in dataset_dict["alex_efficiency"]:
                    dataset_dict["alex_efficiency"][trace_index] = {metric_key: []}

            for trace_index in range(n_traces):

                dd = copy.deepcopy(dataset_dict["dd"][trace_index][metric_key])
                da = copy.deepcopy(dataset_dict["da"][trace_index][metric_key])

                if background_metric_key != None:

                    dd_bg = dataset_dict["dd"][trace_index][background_metric_key].copy()
                    da_bg = dataset_dict["da"][trace_index][background_metric_key].copy()

                    dd = dd - dd_bg
                    da = da - da_bg

                if efficiency_offset:
                    max_value = np.max([np.max(dd), np.max(da)])
                    da = da + max_value
                    dd = dd + max_value

                efficiency = da / ((gamma_correction * dd) + da)
                efficiency = np.array(efficiency)

                if clip_data:
                    efficiency = np.clip(efficiency, 0, 1)

                dataset_dict["alex_efficiency"][trace_index][metric_key] = efficiency

                if progress_callback is not None:
                    progress = int(100 * trace_index / n_traces)
                    progress_callback.emit(progress)

            self.traces_dict[dataset_name] = dataset_dict



        except:
            print(traceback.format_exc())
            pass

    def sort_plot_channels(self, plot_channels):

        try:

            reference_list = ["donor", "acceptor", "fret_efficiency",
                              "dd", "aa", "da", "ad","alex_efficiency",]

            order = {key: i for i, key in enumerate(reference_list)}

            # Sort the actual list based on the order defined in the reference list
            sorted_list = sorted(plot_channels, key=lambda x: order.get(x, float('inf')))

        except:
            pass

        return sorted_list

    def populate_plot_dict(self, progress_callback=None):

        try:

            dataset_name = self.plot_data.currentText()
            channel_name = self.plot_channel.currentText()
            metric_name = self.plot_metric.currentText()
            background_mode = self.plot_background_mode.currentText()

            if hasattr(self, "plot_show_dict") == False:
                self.plot_show_dict = {}

            metric_key = self.get_dict_key(self.metric_dict, metric_name)

            if background_mode != "None":
                key_modifier = self.get_dict_key(self.background_dict, background_mode)
                background_metric_key = metric_key + key_modifier
            else:
                background_metric_key = None

            if dataset_name == "All Datasets":
                plot_datasets = self.traces_dict.keys()
            else:
                plot_datasets = [dataset_name]

            if channel_name == "All Channels":
                plot_channels = [channel for dataset_dict in self.traces_dict.values() for channel in dataset_dict.keys()]
                plot_channels = list(set(plot_channels))
                plot_channels = [channel for channel in plot_channels if "efficiency" not in channel.lower()]
                iteration_channel = plot_channels[0]
                if set(["dd","da"]).issubset(plot_channels):
                    plot_channels.append("alex_efficiency")
                if set(["donor","acceptor"]).issubset(plot_channels):
                    plot_channels.append("fret_efficiency")
            elif channel_name == "ALEX Data":
                plot_channels = ["dd", "da", "ad", "aa"]
                iteration_channel = "aa"
            elif channel_name == "ALEX Efficiency":
                plot_channels = ["alex_efficiency"]
                iteration_channel = "aa"
            elif channel_name == "ALEX Data + Efficiency":
                plot_channels = ["dd", "da", "ad", "aa", "alex_efficiency"]
                iteration_channel = "aa"
            elif channel_name == "FRET Data":
                plot_channels = ["donor", "acceptor"]
                iteration_channel = "donor"
            elif channel_name == "FRET Efficiency":
                plot_channels = ["fret_efficiency"]
                iteration_channel = "donor"
            elif channel_name == "FRET Data + Efficiency":
                plot_channels = ["donor", "acceptor", "fret_efficiency"]
                iteration_channel = "donor"
            else:
                plot_channels = [channel_name.lower()]
                iteration_channel = channel_name.lower()

            plot_channels = self.sort_plot_channels(plot_channels)

            n_iterations = 0
            for dataset in plot_datasets:
                if dataset in self.traces_dict:
                    for channel in plot_channels:
                        if channel in self.traces_dict[dataset]:
                            n_iterations += len(self.traces_dict[dataset][channel].copy())


            iter = 0

            plot_dict = {}

            for dataset_name in plot_datasets:

                if channel_name == "All Channels" or "efficiency" in channel_name.lower():
                    dataset_channels = self.traces_dict[dataset_name].keys()
                    if set(["dd", "da"]).issubset(dataset_channels):

                        self.compute_alex_efficiency(dataset_name, metric_key,
                            background_metric_key, progress_callback,
                            clip_data=True)

                    elif set(["donor", "acceptor"]).issubset(dataset_channels):

                        self.compute_fret_efficiency(dataset_name, metric_key,
                            background_metric_key, progress_callback,
                            clip_data=True)

                for channel_index, channel in enumerate(plot_channels):

                    if channel in self.traces_dict[dataset_name].keys():

                        channel_dict = self.traces_dict[dataset_name][channel].copy()
                        for trace_index, trace_dict in channel_dict.items():

                            data = np.array(trace_dict[metric_key].copy())

                            if "efficiency" not in channel.lower():
                                if background_mode != "None":
                                    background = np.array(trace_dict[background_metric_key].copy())
                                    # background = gaussian_filter1d(background, 10)
                                    data = data - background
                                bleach_index = trace_dict["bleach_index"]
                                donor_bleach_index = trace_dict["donor_bleach_index"]
                                acceptor_bleach_index = trace_dict["acceptor_bleach_index"]

                            if channel in ["dd", "da", "ad", "aa"]:
                                label = f"{channel.upper()} [{metric_name}]"
                            elif channel == "alex_efficiency":
                                label = f"ALEX Efficiency [{metric_name}]"
                            elif channel == "fret_efficiency":
                                label = f"FRET Efficiency [{metric_name}]"
                            else:
                                label = f"{channel.capitalize()} [{metric_name}]"

                            if dataset_name not in plot_dict.keys():
                                plot_dict[dataset_name] = {}
                            if trace_index not in plot_dict[dataset_name].keys():
                                plot_dict[dataset_name][trace_index] = {"labels": [], "data": [],
                                                                        "channels": [],
                                                                        "bleach_index": None,
                                                                        "donor_bleach_index": None,
                                                                        "acceptor_bleach_index": None
                                                                        }

                            plot_dict[dataset_name][trace_index]["labels"].append(label)
                            plot_dict[dataset_name][trace_index]["data"].append(data)
                            plot_dict[dataset_name][trace_index]["channels"].append(channel)

                            if "efficiency" not in channel:
                                plot_dict[dataset_name][trace_index]["bleach_index"] = bleach_index
                                plot_dict[dataset_name][trace_index]["donor_bleach_index"] = donor_bleach_index
                                plot_dict[dataset_name][trace_index]["acceptor_bleach_index"] = acceptor_bleach_index

                            if label not in self.plot_show_dict.keys():
                                label = label.replace("Show: ", "")
                                label = re.sub(r'\[.*?\]', '', label)
                                if label not in self.plot_show_dict.keys():
                                    self.plot_show_dict[label] = True

                            iter += 1

                            if progress_callback is not None:
                                progress = int((iter/n_iterations) * 100)
                                progress_callback.emit(progress)

            self.plot_dict = plot_dict

        except:
            print(traceback.format_exc())
            pass

    def initialize_plot(self):

        try:
            if hasattr(self, "traces_dict"):

                if self.traces_dict != {}:

                    dataset_name = self.plot_data.currentText()
                    channel_name = self.plot_channel.currentText()
                    metric_name = self.plot_metric.currentText()

                    if dataset_name != "" and channel_name != "" and metric_name != "":

                        if self.updating_plot_combos == False:

                            self.plot_localisation_number.setEnabled(False)
                            self.plot_data.setEnabled(False)
                            self.plot_channel.setEnabled(False)
                            self.plot_metric.setEnabled(False)
                            self.split_plots.setEnabled(False)
                            self.normalise_plots.setEnabled(False)

                            self.populate_plot_dict()
                            self.create_plot_checkboxes()
                            self.update_plot_layout()
                            self.plot_traces()

                            self.plot_localisation_number.setEnabled(True)
                            self.plot_data.setEnabled(True)
                            self.plot_channel.setEnabled(True)
                            self.plot_metric.setEnabled(True)
                            self.split_plots.setEnabled(True)
                            self.normalise_plots.setEnabled(True)

        except:
            print(traceback.format_exc())
            pass


    def create_plot_checkboxes(self):

        try:

            grid_layout = self.traces_channel_selection_layout

            channel_list = []
            label_list = []

            for dataset_name, dataset_dict in self.plot_dict.items():

                first_index = list(dataset_dict.keys())[0]

                for label in dataset_dict[first_index]["labels"]:
                    label_list.append(label)
                for channel in dataset_dict[first_index]["channels"]:
                    channel_list.append(channel)

            for i in range(grid_layout.count()):
                item = grid_layout.itemAt(i)
                if item is not None:
                    checkbox = item.widget()
                    checkbox_label = checkbox.text()
                    if isinstance(checkbox, QCheckBox):
                        if checkbox_label not in label_list:
                            checkbox.hide()

            self.repaint()

            if len(channel_list) > 1:
                for col_index, (channel, label) in enumerate(zip(channel_list,label_list)):
                    check_box_name = f"plot_show_{channel}"
                    check_box_label = f"{label}"

                    check_box_label = check_box_label.replace("Show: ","")
                    check_box_label = re.sub(r'\[.*?\]', '', check_box_label)

                    if hasattr(self, check_box_name):
                        check_box = getattr(self, check_box_name)
                        check_box.setText(check_box_label)
                        check_box.show()

                    else:
                        setattr(self, check_box_name, QCheckBox(check_box_label))
                        check_box = getattr(self, check_box_name)

                        check_box.blockSignals(True)
                        check_box.setChecked(True)
                        check_box.blockSignals(False)

                        check_box.stateChanged.connect(self.plot_checkbox_event)

                        self.traces_channel_selection_layout.addWidget(check_box, 0, col_index)

        except:
            print(traceback.format_exc())
            pass

    def plot_checkbox_event(self, event):

        try:

            grid_layout = self.traces_channel_selection_layout

            for i in range(grid_layout.count()):
                item = grid_layout.itemAt(i)
                widget = item.widget()
                if isinstance(widget, QCheckBox):
                    label = widget.text()
                    state = widget.isChecked()

                    label = label.replace("Show: ","")
                    label = re.sub(r'\[.*?\]', '', label)

                    self.plot_show_dict[label] = state

            self.update_plot_layout()
            self.plot_traces()

        except:
            print(traceback.format_exc())
            pass


    def check_efficiency_graph(self, input_string):
        pattern = r"FRET Data \+ Efficiency|ALEX Data \+ Efficiency"
        return re.search(pattern, input_string) is not None

    def update_plot_layout(self):

        try:

            self.plot_grid = {}

            self.graph_canvas.clear()

            split = self.split_plots.isChecked()
            plot_mode = self.plot_channel.currentText()

            efficiency_plot = False

            n_traces = []

            for plot_index, (dataset_name, dataset_dict) in enumerate(self.plot_dict.items()):

                plot_labels = []
                for label in dataset_dict[0]["labels"]:
                    plot_show_label = re.sub(r'\[.*?\]', '', label)
                    if plot_show_label in self.plot_show_dict:
                        if self.plot_show_dict[plot_show_label] == True:
                            plot_labels.append(label)

                if len(plot_labels) > 0:

                    n_plot_lines = len(plot_labels)
                    n_traces.append(len(dataset_dict))

                    sub_plots = []

                    if "Efficiency" in str(plot_labels) and split == False and n_plot_lines > 1:

                        layout = pg.GraphicsLayout()
                        self.graph_canvas.addItem(layout, row=plot_index, col=0)

                        for line_index in range(2):
                            p = CustomPlot()

                            layout.addItem(p, row=line_index, col=0)

                            if line_index != 1:
                                p.hideAxis('bottom')

                            sub_plots.append(p)

                        for j in range(1, len(sub_plots)):
                            sub_plots[j].setXLink(sub_plots[0])

                        efficiency_plot = True

                        top_plot = sub_plots[0]
                        bottom_plot = sub_plots[1]

                        sub_plots = [top_plot]*(n_plot_lines-1) + [bottom_plot]

                    elif split == True and n_plot_lines > 1:

                        layout = pg.GraphicsLayout()
                        self.graph_canvas.addItem(layout, row=plot_index, col=0)

                        for line_index in range(n_plot_lines):
                            p = CustomPlot()

                            layout.addItem(p, row=line_index, col=0)

                            if line_index != n_plot_lines - 1:
                                p.hideAxis('bottom')

                            sub_plots.append(p)

                        for j in range(1, len(sub_plots)):
                            sub_plots[j].setXLink(sub_plots[0])

                    else:
                        layout = self.graph_canvas

                        p = CustomPlot()

                        p.hideAxis('top')
                        p.hideAxis('right')

                        layout.addItem(p, row=plot_index, col=0)

                        for line_index in enumerate(plot_labels):
                            sub_plots.append(p)

                    plot_lines = []
                    plot_lines_labels = []

                    for axes_index, plot in enumerate(sub_plots):

                        line_label = plot_labels[axes_index]
                        line_format = pg.mkPen(color=100 + axes_index * 100, width=2)
                        plot_line = plot.plot(np.zeros(10), pen=line_format, name=line_label)
                        plot.enableAutoRange()
                        plot.autoRange()

                        legend = plot.legend
                        legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

                        plot_details = f"{dataset_name}"

                        if axes_index == 0:
                            plot.setTitle(plot_details)
                            title_plot = plot

                        plotmeta = plot.metadata
                        plotmeta[axes_index] = {"plot_dataset": dataset_name, "line_label": line_label}

                        plot_lines.append(plot_line)
                        plot_lines_labels.append(line_label)

                        self.plot_grid[plot_index] = {
                            "sub_axes": sub_plots,
                            "title_plot": title_plot,
                            "plot_lines": plot_lines,
                            "plot_dataset": dataset_name,
                            "plot_index": plot_index,
                            "n_plot_lines": len(plot_lines),
                            "split": split,
                            "plot_lines_labels": plot_lines_labels,
                            "efficiency_plot": efficiency_plot,
                            }

            if len(n_traces) > 0:

                n_traces = max(n_traces)
                self.plot_localisation_number = self.gui.plot_localisation_number
                self.plot_localisation_number.setMaximum(n_traces-1)

                plot_list = []
                for plot_index, grid in enumerate(self.plot_grid.values()):
                    sub_axes = grid["sub_axes"]
                    sub_plots = []
                    for plot in sub_axes:
                        sub_plots.append(plot)
                        plot_list.append(plot)
                for i in range(1, len(plot_list)):
                    plot_list[i].setXLink(plot_list[0])
                plot.getViewBox().sigXRangeChanged.connect(lambda: auto_scale_y(plot_list))

        except:
            print(traceback.format_exc())
            pass

        return self.plot_grid

    def get_loc_coords(self, localisation_number):

        try:

            if hasattr(self, "image_layer"):

                localisations = self.localisation_dict["bounding_boxes"]["localisations"]
                box_size = self.localisation_dict["bounding_boxes"]["box_size"]
                image_shape = self.image_layer.data.shape

                loc = localisations[localisation_number]

                locX, locY = loc.x, loc.y

                centre = (0, locY, locX)

                x1 = locX - box_size
                x2 = locX + box_size
                y1 = locY - box_size
                y2 = locY + box_size

                zoom = min((image_shape[0] / (y2 - y1)), (image_shape[1] / (x2 - x1)))*2

                self.viewer.camera.center = centre
                self.viewer.camera.zoom = zoom

        except:
            print(traceback.format_exc())
            pass

    def plot_traces(self, update=False):

        try:

            if hasattr(self, "plot_grid") == False:
                self.update_plot_layout()

            if self.plot_grid != {}:

                localisation_number = self.plot_localisation_number.value()

                if self.focus_on_bbox.isChecked() == True:
                    self.get_loc_coords(localisation_number)

                for plot_index, grid in enumerate(self.plot_grid.values()):

                    plot_dataset = grid["plot_dataset"]
                    sub_axes = grid["sub_axes"]
                    plot_lines = grid["plot_lines"]
                    plot_lines_labels = grid["plot_lines_labels"]

                    plot_details = f"{plot_dataset} - N:{localisation_number}"

                    plot_ranges = {"xRange": [0, 100], "yRange": [0, 100]}
                    for line_index, (plot, line, plot_label) in enumerate(zip(sub_axes, plot_lines, plot_lines_labels)):

                        if line_index == 0:
                            plot.setTitle(plot_details)

                        data_index = self.plot_dict[plot_dataset][localisation_number]["labels"].index(plot_label)
                        data = self.plot_dict[plot_dataset][localisation_number]["data"][data_index]

                        if self.normalise_plots.isChecked() and "efficiency" not in plot_label.lower():
                            data = (data - np.min(data)) / (np.max(data) - np.min(data))

                        plot_line = plot_lines[line_index]
                        plot_line.setData(data)

                        if plot_ranges["xRange"][1] < len(data):
                            plot_ranges["xRange"][1] = len(data)
                        if plot_ranges["yRange"][1] < np.max(data):
                            plot_ranges["yRange"][1] = np.max(data)
                        if plot_ranges["yRange"][0] > np.min(data):
                            plot_ranges["yRange"][0] = np.min(data)
                        if plot_ranges["xRange"][0] > 0:
                            plot_ranges["xRange"][0] = 0

                        for line_index, (plot, line, plot_label) in enumerate(zip(sub_axes, plot_lines, plot_lines_labels)):

                            plot.setXRange(min=plot_ranges["xRange"][0],max=plot_ranges["xRange"][1])
                            plot.enableAutoRange(axis="y")

        except:
            print(traceback.format_exc())
            pass













def auto_scale_y(sub_plots):

    try:

        for p in sub_plots:
            data_items = p.listDataItems()

            if not data_items:
                return

            y_min = np.inf
            y_max = -np.inf

            # Get the current x-range of the plot
            plot_x_min, plot_x_max = p.getViewBox().viewRange()[0]

            for index, item in enumerate(data_items):
                if item.name() != "hmm_mean":

                    y_data = item.yData
                    x_data = item.xData

                    # Get the indices of y_data that lies within the current x-range
                    idx = np.where((x_data >= plot_x_min) & (x_data <= plot_x_max))

                    if len(idx[0]) > 0:  # If there's any data within the x-range
                        y_min = min(y_min, y_data[idx].min())
                        y_max = max(y_max, y_data[idx].max())

                    if plot_x_min < 0:
                        x_min = 0
                    else:
                        x_min = plot_x_min

                    if plot_x_max > x_data.max():
                        x_max = x_data.max()
                    else:
                        x_max = plot_x_max

            p.getViewBox().setYRange(y_min, y_max, padding=0)
            p.getViewBox().setXRange(x_min, x_max, padding=0)

    except:
        pass


class CustomPlot(pg.PlotItem):

    def __init__(self, title="", colour="", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.metadata = {}

        self.setMenuEnabled(False)
        self.symbolSize = 100

        legend = self.addLegend(offset=(10, 10))
        legend.setBrush('w')
        legend.setLabelTextSize("8pt")
        self.hideAxis('top')
        self.hideAxis('right')
        self.getAxis('left').setWidth(30)

        self.title = title
        self.colour = colour

        if self.title != "":
            self.setLabel('top', text=title, size="3pt", color=colour)

    def setMetadata(self, metadata_dict):
        self.metadata = metadata_dict

    def getMetadata(self):
        return self.metadata

    def enableAutoRange(self, axis='both'):
        """
        Enables automatic ranging for the specified axis.
        :param axis: 'x', 'y', or 'both' to specify which axis to auto-range.
        """
        if axis == 'x':
            super().enableAutoRange(axis=pg.ViewBox.XAxis)
        elif axis == 'y':
            super().enableAutoRange(axis=pg.ViewBox.YAxis)
        else:
            super().enableAutoRange(axis=pg.ViewBox.XYAxes)


class CustomPyQTGraphWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.frame_position_memory = {}
        self.frame_position = None

def mousePressEvent(self, event):

    if hasattr(self.parent, "plot_grid"):

        if event.modifiers() & Qt.ControlModifier:

            xpos = self.get_event_x_postion(event, mode="click")

        elif event.modifiers() & Qt.AltModifier:

            xpos = self.get_event_x_postion(event, mode="click")

        super().mousePressEvent(event)  # Process the event further

def keyPressEvent(self, event):

    if hasattr(self.parent, "plot_grid"):

        pass

        super().keyPressEvent(event)  # Process the event further

def get_event_x_postion(self, event,  mode="click"):

    self.xpos = None

    if hasattr(self.parent, "plot_grid"):

        if mode == "click":
            pos = event.pos()
            self.scene_pos = self.mapToScene(pos)
        else:
            pos = QCursor.pos()
            self.scene_pos = self.mapFromGlobal(pos)

        # Iterate over all plots
        plot_grid = self.parent.plot_grid

        for plot_index, grid in enumerate(plot_grid.values()):
            sub_axes = grid["sub_axes"]

            for axes_index in range(len(sub_axes)):
                plot = sub_axes[axes_index]

                viewbox = plot.vb
                plot_coords = viewbox.mapSceneToView(self.scene_pos)

        self.xpos = plot_coords.x()

    return self.xpos
