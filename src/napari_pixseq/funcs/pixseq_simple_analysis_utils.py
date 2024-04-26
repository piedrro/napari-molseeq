import traceback
from scipy.ndimage import map_coordinates
import numpy as np
import matplotlib.pyplot as plt

class _simple_analysis_utils:

    def shapes_layer_updated(self, event):
        try:
            if event.action in ["added", "changed", "removed"]:

                shapes_layer = self.viewer.layers["Shapes"]
                shapes = shapes_layer.data

                if len(shapes) > 0:

                    shape_type = shapes_layer.shape_type[-1]

                    if shapes_layer.ndim == 3:

                        if self.verbose:
                            print("reformatting shapes to ndim=2")

                        shapes = shapes_layer.data.copy()
                        shapes = [shape[:,-2:] for shape in shapes]
                        shapes_layer.data = []
                        shapes_layer.add(shapes, shape_type=shape_type)

                    if shape_type == "line":
                        self.simple_plot_mode.setCurrentIndex(0)

                    if shape_type == "rectangle":
                        self.simple_plot_mode.setCurrentIndex(1)

                self.draw_line_plot()

        except:
            print(traceback.format_exc())

    def get_plot_data(self):
        plot_dataset = {}

        try:
            layer_names = [layer.name for layer in self.viewer.layers]

            if "Shapes" in layer_names:
                shapes_layer = self.viewer.layers["Shapes"]
                shapes = shapes_layer.data.copy()
                shape_types = shapes_layer.shape_type.copy()

                plot_mode = self.simple_plot_mode.currentIndex()
                dataset = self.simple_plot_dataset.currentText()

                current_frame = self.viewer.dims.current_step[0]

                for channel in self.dataset_dict[dataset].keys():
                    if channel not in plot_dataset:
                        plot_dataset[channel] = {}

                    for shape_index, (shape, shape_type) in enumerate(zip(shapes, shape_types)):
                        if shape_type == "line" and plot_mode == 0:
                            if shape.shape[-1] == 3:
                                dat = shape[:, 1:]
                            else:
                                dat = shape

                            [[x1, y1], [x2, y2]] = dat

                            x1, y1 = int(x1), int(y1)
                            x2, y2 = int(x2), int(y2)

                            num = int(np.hypot(x2 - x1, y2 - y1))

                            img = self.dataset_dict[dataset][channel]["data"][current_frame]

                            x, y = np.linspace(x1, x2, num), np.linspace(y1, y2, num)
                            coords = np.vstack((x, y))

                            line_profile = map_coordinates(img, coords, order=1, mode="nearest")

                            plot_dataset[channel][shape_index] = line_profile

                        if shape_type == "rectangle" and plot_mode == 1:
                            if shape.shape[-1] == 3:
                                dat = shape[:, 1:]
                            else:
                                dat = shape

                            x1, y1, x2, y2 = (dat[0, 1], dat[0, 0], dat[2, 1], dat[2, 0],)
                            x1, y1 = int(x1), int(y1)
                            x2, y2 = int(x2), int(y2)

                            img = self.dataset_dict[dataset][channel]["data"]

                            box_data = img[:, y1:y2, x1:x2]

                            line_profile = np.mean(box_data, axis=(1, 2))

                            plot_dataset[channel][shape_index] = line_profile

                    if set(["donor", "acceptor"]).issubset(plot_dataset.keys()):
                        plot_dataset = self.calculate_simple_plot_fret(plot_dataset)

        except:
            print(traceback.format_exc())

        return plot_dataset

    def calculate_simple_plot_fret(self, plot_dataset):

        try:
            donor_data = plot_dataset["donor"]
            acceptor_data = plot_dataset["acceptor"]

            for shape_index, (donor, acceptor) in enumerate(zip(donor_data.values(), acceptor_data.values())):
                if "fret" not in plot_dataset.keys():
                    plot_dataset["fret"] = {}

                fret = acceptor / (donor + acceptor)
                plot_dataset["fret"][shape_index] = fret

        except:
            print(traceback.format_exc())

        return plot_dataset

    def draw_line_plot(self):
        try:
            plot_mode = self.simple_plot_mode.currentIndex()
            plot_channel = self.simple_plot_channel.currentText()

            if "efficiency" in plot_channel.lower():
                plot_channel = "fret"

            plot_dataset = self.get_plot_data()

            self.simple_graph_canvas.clear()

            if plot_channel.lower() in plot_dataset.keys():
                plot_data = plot_dataset[plot_channel.lower()]

                if plot_data != {}:
                    if plot_mode == 0:
                        plot_title = "Line profile(s)"
                    if plot_mode == 1:
                        plot_title = "Single Molecule Time Series"

                    ax = self.simple_graph_canvas.addPlot(title=plot_title)

                    for data in plot_data.values():
                        ax.plot(data, pen=(255, 0, 0))

                    plt.show()
        except:
            print(traceback.format_exc())