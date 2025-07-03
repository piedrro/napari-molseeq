from qtpy.QtCore import QThreadPool
from qtpy.QtWidgets import (QWidget,QVBoxLayout)

import numpy as np
import traceback
from multiprocessing import Manager
from functools import partial
import matplotlib.colors as mcolors
from napari.utils.notifications import show_info

from molseeq.GUI.gui import Ui_Frame as gui
from molseeq.funcs.utils_compute import _utils_compute
from molseeq.funcs.undrift_utils import _undrift_utils
from molseeq.funcs.picasso_detect import _picasso_detect_utils
from molseeq.funcs.loc_utils import _loc_utils
from molseeq.funcs.import_utils import _import_utils
from molseeq.funcs.events import _events_utils
from molseeq.funcs.export_images_utils import _export_images_utils
from molseeq.funcs.transform_utils import _tranform_utils
from molseeq.funcs.trace_compute_utils import _trace_compute_utils
from molseeq.funcs.plot_utils import _plot_utils, CustomPyQTGraphWidget
from molseeq.funcs.align_utils import _align_utils
from molseeq.funcs.export_traces_utils import _export_traces_utils
from molseeq.funcs.colocalize_utils import _utils_colocalize
from molseeq.funcs.temporal_filtering import _utils_temporal_filtering
from molseeq.funcs.cluster_utils import _cluster_utils
from molseeq.funcs.simple_analysis_utils import _simple_analysis_utils
from molseeq.funcs.filter_utils import _filter_utils
from molseeq.funcs.tracking_utils import _tracking_utils

import napari


class QWidget(QWidget, gui,
    _undrift_utils, _picasso_detect_utils,
    _import_utils, _events_utils, _export_images_utils,
    _tranform_utils, _trace_compute_utils, _plot_utils,
    _align_utils, _loc_utils, _export_traces_utils,
    _utils_colocalize, _utils_temporal_filtering, _utils_compute,
    _cluster_utils, _simple_analysis_utils,
    _filter_utils, _tracking_utils,):

    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()

        self.viewer = viewer

        #create UI
        self.gui = gui()
        self.gui.setupUi(self)

        #initialise graph PyQtGraph canvases
        self.gui.graph_container.setLayout(QVBoxLayout())
        self.graph_canvas = CustomPyQTGraphWidget(self)
        self.gui.graph_container.layout().addWidget(self.graph_canvas)

        self.gui.simple_graph_container.setLayout(QVBoxLayout())
        self.simple_graph_canvas = CustomPyQTGraphWidget(self)
        self.gui.simple_graph_container.layout().addWidget(self.simple_graph_canvas)

        self.gui.filter_graph_container.setLayout(QVBoxLayout())
        self.filter_graph_canvas = CustomPyQTGraphWidget(self)
        self.gui.filter_graph_container.layout().addWidget(self.filter_graph_canvas)

        #register events
        self.register_events()

        #initialise variables
        self.dataset_dict = {}
        self.traces_dict = {}
        self.plot_dict = {}
        self.contrast_dict = {}
        self.localisation_dict = {"bounding_boxes": {}, "localisations": {}}
        self.metric_dict = {"spot_mean": "Mean", "spot_median": "Median", "spot_sum": "Sum", "spot_max": "Maximum",
                            "spot_std": "std", "spot_photons": "Picasso Photons", }

        self.background_dict = {"None":"None",
                                "_local_bg": "Local Background",
                                "_masked_local_bg": "Masked Local Background",
                                "_global_bg": "Global Background",
                                "_masked_global_bg": "Masked Global Background",
                                "_local_bg": "Local Background",
                                "spot_lsp_bg": "LSP Background",
                                }
        self.active_dataset = None
        self.active_channel = None
        self.verbose = False
        self.worker = None
        self.multiprocessing_active = False
        self.transform_matrix = None

        #create threadpool and stop event
        self.threadpool = QThreadPool()
        manager = Manager()
        self.stop_event = manager.Event()


        self.update_import_options()
        self.update_import_append_options()

        self.check_gpufit_availibility()

        self.widget_notifications = True

        from molseeq.__init__ import __version__ as version
        print(f"napari-molseeq version: {version}")

    def register_events(self):

        self.gui.molseeq_import.clicked.connect(self.molseeq_import_data)
        self.gui.molseeq_import_mode.currentIndexChanged.connect(self.update_import_options)
        self.gui.molseeq_update_dataset_name.clicked.connect(self.update_dataset_name)
        self.gui.molseeq_delete_dataset.clicked.connect(self.delete_dataset)
        self.gui.molseeq_update_labels.clicked.connect(self.update_nucleotide)

        self.gui.import_picasso.clicked.connect(self.import_picaaso_localisations)

        self.gui.picasso_detect.clicked.connect(partial(self.molseeq_picasso, detect = True, fit=False))
        self.gui.picasso_fit.clicked.connect(partial(self.molseeq_picasso, detect = False, fit=True))
        self.gui.picasso_detectfit.clicked.connect(partial(self.molseeq_picasso, detect=True, fit=True))
        self.gui.cluster_localisations.clicked.connect(self.molseeq_cluster_localisations)
        self.gui.dbscan_remove_overlapping = self.gui.dbscan_remove_overlapping

        self.gui.picasso_render.clicked.connect(self.initialise_picasso_render)

        self.gui.molseeq_dataset_selector.currentIndexChanged.connect(self.update_channel_select_buttons)
        self.gui.molseeq_dataset_selector.currentIndexChanged.connect(partial(self.update_active_image,
            dataset = self.gui.molseeq_dataset_selector.currentText()))

        self.gui.picasso_undrift.clicked.connect(self.undrift_images)

        self.gui.molseeq_align_datasets.clicked.connect(self.align_datasets)
        self.gui.align_reference_dataset.currentIndexChanged.connect(self.update_align_reference_channel)

        self.gui.molseeq_import_tform.clicked.connect(self.import_transform_matrix)
        self.gui.molseeq_compute_tform.clicked.connect(self.compute_transform_matrix)
        self.gui.molseeq_apply_tform.clicked.connect(self.apply_transform_matrix)

        self.gui.picasso_detect_mode.currentIndexChanged.connect(self.update_picasso_options)

        self.gui.molseeq_export_data.clicked.connect(self.export_data)
        self.gui.export_dataset.currentIndexChanged.connect(self.update_export_options)

        self.gui.molseeq_export_locs.clicked.connect(self.initialise_export_locs)
        self.gui.locs_export_type.currentIndexChanged.connect(self.update_loc_export_options)
        self.gui.locs_export_dataset.currentIndexChanged.connect(self.update_loc_export_options)

        self.gui.molseeq_export_traces.clicked.connect(self.export_traces)
        self.gui.traces_export_dataset.currentIndexChanged.connect(self.populate_export_combos)

        self.viewer.dims.events.current_step.connect(partial(self.draw_localisations, update_vis = False))

        self.gui.compute_traces.clicked.connect(self.molseeq_compute_traces)
        self.gui.traces_visualise_masks.clicked.connect(self.visualise_spot_masks)
        self.gui.traces_visualise_masks.clicked.connect(self.visualise_background_masks)

        self.gui.plot_data.currentIndexChanged.connect(partial(self.update_plot_combos, combo="plot_data"))
        self.gui.plot_channel.currentIndexChanged.connect(partial(self.update_plot_combos, combo="plot_channel"))

        self.gui.plot_data.currentIndexChanged.connect(self.initialize_plot)
        self.gui.plot_channel.currentIndexChanged.connect(self.initialize_plot)
        self.gui.plot_metric.currentIndexChanged.connect(self.initialize_plot)
        self.gui.background_metric.currentIndexChanged.connect(self.initialize_plot)
        self.gui.split_plots.stateChanged.connect(self.initialize_plot)
        self.gui.normalise_plots.stateChanged.connect(self.initialize_plot)
        self.gui.plot_background_mode.currentIndexChanged.connect(self.initialize_plot)
        self.gui.focus_on_bbox.stateChanged.connect(self.initialize_plot)

        self.gui.molseeq_colocalize.clicked.connect(self.molseeq_colocalize_localisations)

        self.gui.plot_localisation_number.valueChanged.connect(lambda: self.update_slider_label("plot_localisation_number"))
        self.gui.plot_localisation_number.valueChanged.connect(partial(self.plot_traces))

        self.gui.filtering_start.clicked.connect(self.molseeq_temporal_filtering)
        self.gui.filtering_datasets.currentIndexChanged.connect(self.update_filtering_channels)

        self.gui.molseeq_append.stateChanged.connect(self.update_import_append_options)

        self.gui.picasso_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="picasso_dataset", channel_selector="picasso_channel"))
        self.gui.undrift_dataset_selector.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="undrift_dataset_selector", channel_selector="undrift_channel_selector"))
        self.gui.cluster_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="cluster_dataset", channel_selector="cluster_channel"))
        self.gui.tform_compute_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="tform_compute_dataset", channel_selector="tform_compute_ref_channel", channel_type="donor"))
        self.gui.tform_compute_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="tform_compute_dataset", channel_selector="tform_compute_target_channel", channel_type="acceptor"))
        self.gui.colo_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="colo_dataset", channel_selector="colo_channel1"))
        self.gui.colo_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="colo_dataset", channel_selector="colo_channel2"))
        self.gui.picasso_filter_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="picasso_filter_dataset", channel_selector="picasso_filter_channel"))
        self.gui.simple_plot_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="simple_plot_dataset", channel_selector="simple_plot_channel", efficiency=True))
        self.gui.picasso_render_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="picasso_render_dataset", channel_selector="picasso_render_channel"))
        self.gui.tracking_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="tracking_dataset", channel_selector="tracking_channel"))

        self.gui.traces_export_mode.currentIndexChanged.connect(self.populate_export_combos)

        self.gui.picasso_filter_dataset.currentIndexChanged.connect(self.update_filter_criterion)
        self.gui.picasso_filter_channel.currentIndexChanged.connect(self.update_filter_criterion)
        self.gui.filter_criterion.currentIndexChanged.connect(self.update_criterion_ranges)
        self.gui.filter_localisations.clicked.connect(self.molseeq_filter_localisations)
        self.gui.picasso_filter_type.currentIndexChanged.connect(self.update_filter_dataset)

        self.viewer.bind_key('Control-D', self.dev_function)

        self.viewer.bind_key('PageUp', self.named_partial(self.increment_active_dataset, key='Up'), overwrite=True)
        self.viewer.bind_key('PageDown', self.named_partial(self.increment_active_dataset, key='Down'), overwrite=True)

        self.viewer.bind_key('Q', self.stop_worker, overwrite=True)

        self.viewer.layers.events.inserted.connect(self.on_add_layer)

        self.gui.simple_plot_mode.currentIndexChanged.connect(self.update_plot_channel)
        self.gui.simple_plot_dataset.currentIndexChanged.connect(self.draw_line_plot)
        self.gui.simple_plot_channel.currentIndexChanged.connect(self.draw_line_plot)
        self.gui.simple_subtract_background.clicked.connect(self.draw_line_plot)

        self.gui.add_line.clicked.connect(lambda: self.draw_shapes(mode="line"))
        self.gui.add_box.clicked.connect(lambda: self.draw_shapes(mode="box"))
        self.gui.add_background.clicked.connect(lambda: self.draw_shapes(mode="background"))

        self.gui.picasso_vis_mode.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))
        self.gui.picasso_vis_mode.currentIndexChanged.connect(partial(self.draw_bounding_boxes, update_vis=True))
        self.gui.picasso_vis_size.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))
        self.gui.picasso_vis_size.currentIndexChanged.connect(partial(self.draw_bounding_boxes, update_vis=True))
        self.gui.picasso_vis_opacity.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))
        self.gui.picasso_vis_opacity.currentIndexChanged.connect(partial(self.draw_bounding_boxes, update_vis=True))
        self.gui.picasso_vis_edge_width.currentIndexChanged.connect(partial(self.draw_localisations, update_vis=True))
        self.gui.picasso_vis_edge_width.currentIndexChanged.connect(partial(self.draw_bounding_boxes, update_vis=True))

        self.gui.link_localisations.clicked.connect(self.initialise_tracking)

        self.gui.dev_verbose.stateChanged.connect(self.toggle_verbose)

        self.viewer.bind_key("a", func=lambda event: self.segmentation_modify_mode(mode="add"), overwrite=True)
        self.viewer.bind_key("e", func=lambda event: self.segmentation_modify_mode(mode="extend"), overwrite=True)
        self.viewer.bind_key("j", func=lambda event: self.segmentation_modify_mode(mode="join"), overwrite=True)
        self.viewer.bind_key("s", func=lambda event: self.segmentation_modify_mode(mode="split"), overwrite=True)
        self.viewer.bind_key("d", func=lambda event: self.segmentation_modify_mode(mode="delete"), overwrite=True)


        # self.viewer.bind_key("e", func=self.segmentation_modify_mode(mode="extend"), overwrite=True)
        # self.viewer.bind_key("j", func=self.segmentation_modify_mode(mode="join"), overwrite=True)
        # self.viewer.bind_key("s", func=self.segmentation_modify_mode(mode="split"), overwrite=True)
        # self.viewer.bind_key("d", func=self.segmentation_modify_mode(mode="delete"), overwrite=True)



    def check_gpufit_availibility(self):

        self.gpufit_available = False

        try:
            from pygpufit import gpufit as gf
            package_installed = True
        except:
            package_installed = False

        if package_installed:

            if not gf.cuda_available():
                print("Pygpufit not available due to missing CUDA")
            else:
                runtime_version, driver_version = gf.get_cuda_version()

                runtime_version = ".".join([str(v) for v in list(runtime_version)])
                driver_version = ".".join([str(v) for v in list(driver_version)])

                if runtime_version != driver_version:
                    print(f"Pygpufit not available due to CUDA version mismatch. "
                          f"Runtime: {runtime_version}, Driver: {driver_version}")

                else:
                    self.gpufit_available = True

        else:
            print("Pygpufit not available due to missing package")
            print("Install pygpufit package into napari-molseeq root directory")

        if self.gpufit_available:
            print("Pygpufit available")
            self.gui.picasso_use_gpufit.setEnabled(True)
        else:
            self.gui.picasso_use_gpufit.setEnabled(False)


    def on_add_layer(self, event):
        if event.value.name == "Shapes":
            layer_list = [layer.name for layer in self.viewer.layers if layer.name != "Shapes"]

            self.shapes_layer = self.viewer.layers["Shapes"]

            self.shapes_layer.events.data.connect(self.shapes_layer_updated)

            self.shapes_layer.current_edge_color = list(mcolors.to_rgb("green"))
            self.shapes_layer.current_face_color = [0, 0, 0, 0]
            self.shapes_layer.current_edge_width = 1


    def toggle_verbose(self):

        if self.gui.dev_verbose.isChecked():
            self.verbose = True
        else:
            self.verbose = False

    def dev_function(self, event):

        print("Dev function called")

        self.check_gpufit_availibility()



    def select_image_layer(self):

        try:
            if hasattr(self, "image_layer"):
                self.viewer.layers.selection.select_only(self.image_layer)
        except:
            print(traceback.format_exc())
            pass

    def add_lsp_localisation(self, position = None):

        try:
            layer_names = [layer.name for layer in self.viewer.layers]

            vis_mode = self.gui.picasso_vis_mode.currentText()
            vis_size = float(self.gui.picasso_vis_size.currentText())
            vis_opacity = float(self.gui.picasso_vis_opacity.currentText())
            vis_edge_width = float(self.gui.picasso_vis_edge_width.currentText())

            if vis_mode.lower() == "square":
                symbol = "square"
            elif vis_mode.lower() == "disk":
                symbol = "disc"
            elif vis_mode.lower() == "x":
                symbol = "cross"

            if position is not None:

                if hasattr(self, "lsp_locs"):

                    distances = np.sqrt(np.sum((self.lsp_locs - np.array(position)) ** 2, axis=1))

                    min_index = np.argmin(distances)
                    min_distance = distances[min_index]

                    if min_distance < vis_size/2:
                        self.lsp_locs.pop(min_index)
                    else:
                        self.lsp_locs.append(position)
                else:
                    self.lsp_locs = [position]

            if hasattr(self, "lsp_locs"):

                lsp_locs = list(self.lsp_locs)

                if "LSP localisations" not in layer_names:
                    self.lsp_layer = self.viewer.add_points(lsp_locs,
                        edge_color="green",
                        ndim=2,
                        face_color=[0, 0, 0,0],
                        opacity=vis_opacity,
                        name="LSP localisations",
                        symbol=symbol,
                        size=vis_size,
                        visible=True,
                        edge_width=vis_edge_width, )

                    self.lsp_layer.mouse_drag_callbacks.append(self._mouse_event)
                    self.lsp_layer.selected_data = []

                else:
                    self.lsp_layer.data = lsp_locs
                    self.lsp_layer.selected_data = []

                self.lsp_layer.refresh()

        except:
            print(traceback.format_exc())
            pass

    def draw_bounding_boxes(self, update_vis=False):

        if hasattr(self, "localisation_dict") and hasattr(self, "active_channel"):

            if hasattr(self, "bbox_layer"):
                show_bboxes = self.bbox_layer.visible
            else:
                show_bboxes = True

            if show_bboxes:

                layer_names = [layer.name for layer in self.viewer.layers]
                loc_dict, n_locs, fitted = self.get_loc_dict(type = "bounding_boxes")

                if n_locs > 0:

                    if self.verbose:
                        print("Drawing bounding_boxes")

                    locs = loc_dict["localisations"].copy()
                    bounding_boxes = np.vstack((locs.y, locs.x)).T.tolist()

                    vis_mode = self.gui.picasso_vis_mode.currentText()
                    vis_size = float(self.gui.picasso_vis_size.currentText())
                    vis_opacity = float(self.gui.picasso_vis_opacity.currentText())
                    vis_edge_width = float(self.gui.picasso_vis_edge_width.currentText())

                    if vis_mode.lower() == "square":
                        symbol = "square"
                    elif vis_mode.lower() == "disk":
                        symbol = "disc"
                    elif vis_mode.lower() == "x":
                        symbol = "cross"

                    if "bounding_boxes" not in layer_names:
                        self.bbox_layer = self.viewer.add_points(
                            bounding_boxes,
                            edge_color="white",
                            ndim=2,
                            face_color=[0,0,0,0],
                            opacity=vis_opacity,
                            name="bounding_boxes",
                            symbol=symbol,
                            size=vis_size,
                            visible=True,
                            edge_width=vis_edge_width,)

                        self.bbox_layer.mouse_drag_callbacks.append(self._mouse_event)
                        self.bbox_layer.events.visible.connect(self.draw_bounding_boxes)

                    else:
                        self.viewer.layers["bounding_boxes"].data = bounding_boxes

                    self.bbox_layer.selected_data = list(range(len(self.bbox_layer.data)))
                    self.bbox_layer.opacity = vis_opacity
                    self.bbox_layer.symbol = symbol
                    self.bbox_layer.size = vis_size
                    self.bbox_layer.edge_width = vis_edge_width
                    self.bbox_layer.edge_color = "white"
                    self.bbox_layer.selected_data = []
                    self.bbox_layer.refresh()

                for layer in layer_names:
                    self.viewer.layers[layer].refresh()


    def draw_localisations(self, update_vis=False):

        remove_localisations = True

        if hasattr(self, "localisation_dict") and hasattr(self, "active_channel"):

            if hasattr(self, "fiducial_layer"):
                show_localisations = self.fiducial_layer.visible
            else:
                show_localisations = True

            if show_localisations:

                layer_names = [layer.name for layer in self.viewer.layers]

                active_frame = self.viewer.dims.current_step[0]

                dataset_name = self.gui.molseeq_dataset_selector.currentText()
                image_channel = self.active_channel

                if image_channel != "" and dataset_name != "":

                    if image_channel.lower() in self.localisation_dict["localisations"][dataset_name].keys():
                        localisation_dict = self.localisation_dict["localisations"][dataset_name][image_channel.lower()].copy()

                        if "localisations" in localisation_dict.keys():

                            locs = localisation_dict["localisations"]

                            if active_frame in locs.frame:

                                frame_locs = locs[locs.frame == active_frame].copy()
                                render_locs = np.vstack((frame_locs.y, frame_locs.x)).T.tolist()

                                vis_mode = self.gui.picasso_vis_mode.currentText()
                                vis_size = float(self.gui.picasso_vis_size.currentText())
                                vis_opacity = float(self.gui.picasso_vis_opacity.currentText())
                                vis_edge_width = float(self.gui.picasso_vis_edge_width.currentText())

                                if vis_mode.lower() == "square":
                                    symbol = "square"
                                elif vis_mode.lower() == "disk":
                                    symbol = "disc"
                                elif vis_mode.lower() == "x":
                                    symbol = "cross"

                                remove_localisations = False

                                if "localisations" not in layer_names:

                                    if self.verbose:
                                        print("Drawing localisations")

                                    self.fiducial_layer = self.viewer.add_points(
                                        render_locs,
                                        ndim=2,
                                        edge_color="red",
                                        face_color=[0,0,0,0],
                                        opacity=vis_opacity,
                                        name="localisations",
                                        symbol=symbol,
                                        size=vis_size,
                                        edge_width=vis_edge_width, )

                                    self.fiducial_layer.mouse_drag_callbacks.append(self._mouse_event)
                                    self.fiducial_layer.events.visible.connect(self.draw_localisations)

                                    update_vis = True

                                else:

                                    if self.verbose:
                                        print("Updating fiducial data")

                                    self.fiducial_layer.data = render_locs
                                    self.fiducial_layer.selected_data = []

                                if update_vis:

                                    if self.verbose:
                                        print("Updating fiducial visualisation settings")

                                    self.fiducial_layer.selected_data = list(range(len(self.fiducial_layer.data)))
                                    self.fiducial_layer.opacity = vis_opacity
                                    self.fiducial_layer.symbol = symbol
                                    self.fiducial_layer.size = vis_size
                                    self.fiducial_layer.edge_width = vis_edge_width
                                    self.fiducial_layer.edge_color = "red"
                                    self.fiducial_layer.selected_data = []
                                    self.fiducial_layer.refresh()



                if remove_localisations:
                    if "localisations" in layer_names:
                        self.viewer.layers["localisations"].data = []

                for layer in layer_names:
                    self.viewer.layers[layer].refresh()


    def get_localisation_centres(self, locs, mode = "localisations"):

        loc_centres = []

        try:

            for loc in locs:
                frame = int(loc.frame)
                if mode == "localisations":
                    loc_centres.append([frame, loc.y, loc.x])
                else:
                    loc_centres.append([loc.y, loc.x])

        except:
            print(traceback.format_exc())

        return loc_centres

    def closeEvent(self):
        print("Closing molSEEQ")
