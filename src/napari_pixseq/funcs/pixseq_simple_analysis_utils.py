import traceback
from scipy.ndimage import map_coordinates
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg

class _simple_analysis_utils:

    def update_plot_channel(self):

        plot_mode = self.gui.simple_plot_mode.currentIndex()
        plot_channel = self.gui.simple_plot_channel.currentText()
        plot_dataset = self.gui.simple_plot_dataset.currentText()

        plot_channels = list(self.dataset_dict[plot_dataset].keys())

        for index, channel in enumerate(plot_channels):
            if channel in ["dd","da","aa","ad"]:
                plot_channels[index] = channel.upper()
            else:
                plot_channels[index] = channel.capitalize()

        if plot_mode in [2, 3]:
            if set(["Donor", "Acceptor"]).issubset(plot_channels):
                plot_channels.extend(["FRET Data", "FRET Efficiency"])
            if set(["DD", "DA"]).issubset(plot_channels):
                plot_channels.extend(["ALEX Data", "ALEX Efficiency"])

        self.gui.simple_plot_channel.blockSignals(True)
        self.gui.simple_plot_channel.clear()
        self.gui.simple_plot_channel.addItems(plot_channels)

        if plot_channel in plot_channels:
            index = plot_channels.index(plot_channel)
            self.gui.simple_plot_channel.setCurrentIndex(index)

        self.gui.simple_plot_channel.blockSignals(False)

        self.draw_line_plot()


    def get_shapes_layer(self):

        layer_names = [layer.name for layer in self.viewer.layers]

        if "Shapes" not in layer_names:
            properties = {"mode": [], "ndim": 2}
            shapes_layer = self.viewer.add_shapes(name="Shapes", face_color="transparent",
                edge_color="red", edge_width=0.1, ndim=2, properties=properties, )
        else:
            shapes_layer = self.viewer.layers["Shapes"]

        return shapes_layer


    def draw_shapes(self, mode="line"):
        try:
            self.shapes_layer = self.get_shapes_layer()

            if mode == "line":
                self.shapes_layer.mode = 'add_line'
                self.shapes_layer.current_edge_color = "red"
                self.shapes_layer.current_properties = {"mode": "line"}
            elif mode == "box":
                self.shapes_layer.mode = 'add_rectangle'
                self.shapes_layer.current_edge_color = "green"
                self.shapes_layer.current_properties = {"mode": "box"}
            elif mode == "background":
                self.shapes_layer.mode = 'add_rectangle'
                self.shapes_layer.current_edge_color = "white"
                self.shapes_layer.current_properties = {"mode": "background"}

            self.shapes_layer.current_face_color = [0, 0, 0, 0]
            self.shapes_layer.current_edge_width = 2

        except:
            print(traceback.format_exc())

    def shapes_layer_updated(self, event):
        try:
            if event.action in ["added"]:
                shapes_layer = self.viewer.layers["Shapes"]
                shapes = shapes_layer.data
                shape_types = shapes_layer.shape_type
                shape_properties = shapes_layer.properties

                if len(shapes) > 0:
                    shape_type = shapes_layer.shape_type[-1]

                    if shapes_layer.ndim == 3:
                        if self.verbose:
                            print("reformatting shapes to ndim=2")

                        shapes = shapes_layer.data.copy()
                        shapes = [shape[:, -2:] for shape in shapes]
                        shapes_layer.data = []
                        shapes_layer.add(shapes, shape_type=shape_type)

                    if event.action == "added":
                        n_shapes = len(shapes)
                        if n_shapes > 1:
                            shape_modes = shape_properties["mode"]

                            delete_list = []

                            for mode in np.unique(shape_modes):
                                delete_indices = np.where(shape_modes == mode)[0][:-1]
                                delete_list.extend(delete_indices)

                            shapes_layer.selected_data = set(delete_list)
                            shapes_layer.remove_selected()

                    if shape_type == "line":
                        self.gui.simple_plot_mode.setCurrentIndex(0)

                    if shape_type == "rectangle":
                        self.gui.simple_plot_mode.setCurrentIndex(2)

            if event.action in ["added", "changed", "removed"]:
                self.draw_line_plot()

        except:
            print(traceback.format_exc())

    def get_box_profile(self, box, dataset, channel):
        x1, y1, x2, y2 = (box[0, 1], box[0, 0], box[2, 1], box[2, 0],)
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        img = self.dataset_dict[dataset][channel]["data"]

        box_data = img[:, y1:y2, x1:x2]

        box_profile = np.mean(box_data, axis=(1, 2))

        return box_profile

    def get_line_profile(self, line, dataset, channel, current_frame):
        [[x1, y1], [x2, y2]] = line

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        num = int(np.hypot(x2 - x1, y2 - y1))

        img = self.dataset_dict[dataset][channel]["data"][current_frame]

        x, y = np.linspace(x1, x2, num), np.linspace(y1, y2, num)
        coords = np.vstack((x, y))

        line_profile = map_coordinates(img, coords, order=1, mode="nearest")

        return line_profile

    def get_plot_data(self):
        plot_data = {}

        try:
            layer_names = [layer.name for layer in self.viewer.layers]

            if "Shapes" in layer_names:
                shapes_layer = self.viewer.layers["Shapes"]
                shapes = shapes_layer.data.copy()
                shape_types = shapes_layer.shape_type.copy()
                shape_modes = shapes_layer.properties["mode"].tolist()

                plot_mode = self.gui.simple_plot_mode.currentIndex()
                plot_channel = self.gui.simple_plot_channel.currentText()
                dataset = self.gui.simple_plot_dataset.currentText()
                subtract_background = self.gui.simple_subtract_background.isChecked()
                current_frame = self.viewer.dims.current_step[0]

                if plot_channel.lower() in ["donor", "acceptor","dd","da","aa","ad"]:
                    plot_channels = [plot_channel.lower()]
                elif plot_channel.lower() in ["fret data", "fret efficiency"]:
                    plot_channels = ["donor", "acceptor"]
                elif plot_channel.lower() in ["alex data", "alex efficiency"]:
                    plot_channels = ["dd", "da"]

                for channel in plot_channels:
                    if plot_mode in [0, 1]:
                        dat = [shape for mode, shape in zip(shape_modes, shapes) if mode == "line"]

                        if len(dat) > 0:
                            line_profile = self.get_line_profile(dat[0], dataset, channel, current_frame)

                            if channel.lower() in ["dd", "da", "aa", "ad"]:
                                plot_data[channel.upper()] = line_profile
                            else:
                                plot_data[channel.capitalize()] = line_profile

                    if plot_mode == 2:
                        box = [shape for mode, shape in zip(shape_modes, shapes) if mode == "box"]
                        background = [shape for mode, shape in zip(shape_modes, shapes) if mode == "background"]

                        if len(box) == 1:
                            box_profile = self.get_box_profile(box[0], dataset, channel)

                            if len(background) == 1 and subtract_background == True:
                                background_profile = self.get_box_profile(background[0], dataset, channel)

                                if channel.lower() in ["dd", "da", "aa", "ad"]:
                                    plot_data[channel.upper()] = box_profile - background_profile
                                else:
                                    plot_data[channel.capitalize()] = box_profile - background_profile

                            else:

                                if channel.lower() in ["dd", "da", "aa", "ad"]:
                                    plot_data[channel.upper()] = box_profile
                                else:
                                    plot_data[channel.capitalize()] = box_profile

                if set(["Donor", "Acceptor"]).issubset(plot_data.keys()) and plot_mode in [2, 3]:
                    if plot_channel.lower() == "fret efficiency":
                        plot_data = self.calculate_simple_efficiency(plot_data, mode="fret")
                        plot_data.pop("Donor")
                        plot_data.pop("Acceptor")
                if set(["DD", "DA"]).issubset(plot_data.keys()) and plot_mode in [2, 3]:
                    if plot_channel.lower() == "alex efficiency":
                        plot_data = self.calculate_simple_efficiency(plot_data, mode="alex")
                        plot_data.pop("DD")
                        plot_data.pop("DA")

        except:
            print(traceback.format_exc())

        return plot_data

    def calculate_simple_efficiency(self, plot_data, mode = "fret"):
        try:
            if mode == "fret":
                donor = plot_data["Donor"]
                acceptor = plot_data["Acceptor"]

                efficiency = acceptor / (donor + acceptor)
                plot_data["FRET Efficiency"] = efficiency

            else:
                dd = plot_data["DD"]
                da = plot_data["DA"]

                efficiency = da / (dd + da)
                plot_data["ALEX Efficiency"] = efficiency

        except:
            print(traceback.format_exc())

        return plot_data

    def draw_line_plot(self):
        try:
            plot_mode = self.gui.simple_plot_mode.currentIndex()
            plot_data = self.get_plot_data()

            self.simple_graph_canvas.clear()

            if plot_mode == 0:
                plot_title = "Line profile(s)"
            if plot_mode == 1:
                plot_title = "Line profile(s)"
            if plot_mode in [2, 3]:
                plot_title = "Single Molecule Time Series"

            ax = self.simple_graph_canvas.addPlot(title=plot_title)

            legend = pg.LegendItem(offset=(-50, 20), brush=(255, 255, 255, 50))
            legend.setParentItem(ax.graphicsItem())

            for label, data in plot_data.items():
                y_data = np.array(data)
                x_data = np.arange(len(y_data))

                if label.lower() in ["donor","dd","ad"]:
                    colour = (255, 0, 0)
                if label.lower() in ["acceptor", "da", "aa"]:
                    colour = (0, 255, 0)
                if label.lower() in ["fret efficiency","alex efficiency"]:
                    colour = (0, 0, 255)

                if plot_mode in [0, 1]:
                    line = ax.plot(x_data, y_data, pen=colour, symbol="o", symbolPen=(255, 255, 255), name=label)
                    legend.addItem(line, label)
                else:
                    line = ax.plot(x_data, y_data, pen=colour, name=label)
                    legend.addItem(line, label)

                if plot_mode == 1:
                    try:
                        fitX, fitY, peak_width = self.fit_custom_gaussian(x_data, y_data)
                        ax.plot(fitX, fitY, pen=(0, 255, 0), name="Gaussian fit")
                        text = pg.TextItem(f"Peak FWHM {peak_width:.2f}", color=(200, 200, 200))
                        ax.addItem(text)

                        x_min, x_max = ax.viewRange()[0]
                        y_min, y_max = ax.viewRange()[1]
                        offset_x = 0.05 * (x_max - x_min)
                        offset_y = 0.05 * (y_max - y_min)

                        text.setPos(x_min + offset_x, y_max - offset_y)
                    except:
                        pass

            plt.show()

        except:
            print(traceback.format_exc())

    def custom_gaussian(self, x, a, b, c, d):
        """ Custom Gaussian model with baseline and peak height parameters. """
        return a + (b - a) * np.exp(-(x - c) ** 2 / (2 * d ** 2))

    def fit_custom_gaussian(self, x_data, y_data, upscale=True):
        """ Fit the custom Gaussian model to 1D data. """
        # Initial guesses:
        # a: Minimum y-data (baseline)
        # b: Maximum y-data
        # c: x value at the maximum y
        # d: Approximate width from data spread

        import numpy as np
        from scipy.optimize import curve_fit

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        initial_guess = [min(y_data), max(y_data), x_data[np.argmax(y_data)], np.std(x_data) / 2]

        # Fit the model
        popt, pcov = curve_fit(self.custom_gaussian, x_data, y_data, p0=initial_guess)

        if upscale:
            fitX = np.linspace(start=min(x_data), stop=max(x_data), num=1000)
        else:
            fitX = x_data

        # Generate the fitted curve
        fitY = self.custom_gaussian(fitX, *popt)

        width = 2 * np.sqrt(2 * np.log(2)) * popt[3]

        return fitX, fitY, width