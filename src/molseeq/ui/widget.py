import sys
import traceback
from functools import partial
from threading import Event

import matplotlib.colors as mcolors
import napari
import numpy as np
from qtpy.QtCore import QThreadPool
from qtpy.QtWidgets import QComboBox, QDockWidget, QLabel, QVBoxLayout
from qtpy.QtWidgets import QWidget as QtWidget

from molseeq import __version__
from molseeq.session import ProjectSession
from molseeq.trace_views import trace_view
from molseeq.ui.channel_transforms import ChannelTransformsMixin
from molseeq.ui.clustering import ClusteringMixin
from molseeq.ui.colocalisation import ColocalisationMixin
from molseeq.ui.dataset_alignment import DatasetAlignmentMixin
from molseeq.ui.detection import DetectionMixin
from molseeq.ui.drift import DriftCorrectionMixin
from molseeq.ui.events import ViewerEventsMixin
from molseeq.ui.gui import Ui_Frame
from molseeq.ui.image_export import ImageExportMixin
from molseeq.ui.localisation_filtering import LocalisationFilteringMixin
from molseeq.ui.localisations import LocalisationsMixin
from molseeq.ui.plots import CustomPyQTGraphWidget, PlottingMixin
from molseeq.ui.recording_import import RecordingImportMixin
from molseeq.ui.temporal_filtering import TemporalFilteringMixin
from molseeq.ui.trace_analysis import TraceAnalysisMixin
from molseeq.ui.trace_export import TraceExportMixin
from molseeq.ui.trace_extraction import TraceExtractionMixin
from molseeq.ui.tracking import TrackingMixin


class QWidget(
    QtWidget,
    DriftCorrectionMixin,
    DetectionMixin,
    RecordingImportMixin,
    ViewerEventsMixin,
    ImageExportMixin,
    ChannelTransformsMixin,
    TraceExtractionMixin,
    PlottingMixin,
    DatasetAlignmentMixin,
    LocalisationsMixin,
    TraceExportMixin,
    ColocalisationMixin,
    TemporalFilteringMixin,
    ClusteringMixin,
    TraceAnalysisMixin,
    LocalisationFilteringMixin,
    TrackingMixin,
):
    """MolSeeq's napari dock widget."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()

        self.viewer = viewer

        #create UI
        self.gui = Ui_Frame()
        self.gui.setupUi(self)
        self.setWindowTitle(f"MolSeeq v{__version__}")
        models = ["Translation", "Rigid", "Similarity", "Affine", "Homography"]
        self.gui.tform_compute_model = QComboBox(self)
        self.gui.tform_compute_model.addItems(models)
        self.gui.tform_compute_model.setCurrentText("Homography")
        self.gui.formLayout_7.addRow("Transform model", self.gui.tform_compute_model)
        self.gui.align_model = QComboBox(self)
        self.gui.align_model.addItems(["Auto", *models])
        self.gui.align_model.setToolTip("Auto chooses a model from the number of matched fiducials.")
        self.gui.formLayout_10.addRow("Transform model", self.gui.align_model)

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
        self.session = ProjectSession()
        self.dataset_dict = self.session.images
        self.plot_dict = {}
        self.contrast_dict = {}
        self.localisation_dict = self.session.localisation_state
        self.traces_dict = {}
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
        self.stop_event = Event()


        self.update_import_options()
        self.update_import_append_options()

        self.check_gpufit_availibility()

        self.widget_notifications = True

    def showEvent(self, event):
        super().showEvent(event)
        dock = self.parentWidget()
        while dock is not None and not isinstance(dock, QDockWidget):
            dock = dock.parentWidget()
        if dock is not None and dock is not getattr(self, "_title_dock", None):
            self._title_dock = dock
            dock.topLevelChanged.connect(self._update_dock_title)
            dock.dockLocationChanged.connect(self._update_dock_title)
        self._update_dock_title()

    def _update_dock_title(self, *_):
        dock = getattr(self, "_title_dock", None)
        if dock is not None:
            dock.setWindowTitle(self.windowTitle())
            title = getattr(dock.titleBarWidget(), "title", None)
            if isinstance(title, QLabel):
                title.setText(self.windowTitle())

    def bounding_box_token(self):
        return self.session.bounding_box_token()

    @property
    def traces_dict(self):
        sites = self.bounding_box_token()
        if self._trace_sites != sites:
            self._trace_sites = sites
        if self._trace_revision != self.session.trace_token():
            self._traces = trace_view(self.session)
            self._trace_revision = self.session.trace_token()
            self.compute_photo_bleaching()
        return self._traces

    @traces_dict.setter
    def traces_dict(self, traces):
        self._traces = traces
        self._trace_sites = self.bounding_box_token()
        self._trace_revision = self.session.trace_token()

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

        self.gui.picasso_undrift.clicked.connect(self.rcc_undrift)
        self.gui.aim_undrift.clicked.connect(self.aim_undrift)

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
        if getattr(sys, "frozen", False):
            self.gui.picasso_use_gpufit.setEnabled(False)
            return

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
        if not hasattr(self, "localisation_dict"):
            return
        entry = self.localisation_dict["bounding_boxes"]
        rows = entry.get("localisations")
        points = np.empty((0, 2)) if rows is None else np.column_stack((rows.y, rows.x))
        if "bounding_boxes" not in self.viewer.layers:
            if not len(points):
                return
            self.bbox_layer = self.viewer.add_points(
                points, name="bounding_boxes", ndim=2, border_color="white",
                face_color=[0, 0, 0, 0],
            )
            self.bbox_layer.mouse_drag_callbacks.append(self._mouse_event)
            self.bbox_layer.events.visible.connect(self.draw_bounding_boxes)
            self.bbox_layer.events.data.connect(self._bounding_box_layer_changed)
        else:
            self.bbox_layer = self.viewer.layers["bounding_boxes"]
        with self.bbox_layer.events.data.blocker():
            self.bbox_layer.data = points
            self.bbox_layer.features = {"bounding_box_id": list(self.session.bounding_box_ids())}
        mode = self.gui.picasso_vis_mode.currentText().lower()
        self.bbox_layer.symbol = {"square": "square", "disk": "disc", "x": "cross"}[mode]
        self.bbox_layer.size = float(self.gui.picasso_vis_size.currentText())
        self.bbox_layer.opacity = float(self.gui.picasso_vis_opacity.currentText())
        self.bbox_layer.border_width = float(self.gui.picasso_vis_edge_width.currentText())
        self.bbox_layer.border_color = "white"
        self.bbox_layer.selected_data = []
        self.bbox_layer.refresh()

    def _bounding_box_layer_changed(self, event):
        if getattr(event, "action", "changed") not in ("changed", "added", "removed"):
            return
        points = np.asarray(self.bbox_layer.data)
        entry = self.localisation_dict["bounding_boxes"].copy()
        previous = entry.get("localisations")
        dtype = previous.dtype if previous is not None else [("frame", "u4"), ("x", "f4"), ("y", "f4")]
        rows = np.zeros(len(points), dtype=dtype).view(np.recarray)
        old_ids = self.session.bounding_box_ids()
        indexes = {identifier: index for index, identifier in enumerate(old_ids)}
        features = self.bbox_layer.features
        incoming = features["bounding_box_id"].tolist() if "bounding_box_id" in features else [None] * len(points)
        identifiers = []
        seen = set()
        for identifier in incoming:
            if isinstance(identifier, str) and identifier in indexes and identifier not in seen:
                identifiers.append(identifier)
                seen.add(identifier)
            else:
                identifiers.append(None)
        for index, identifier in enumerate(identifiers):
            if identifier is not None:
                rows[index] = previous[indexes[identifier]]
            elif "frame" in rows.dtype.names:
                rows.frame[index] = self.viewer.dims.current_step[0]
        if len(points):
            rows.x, rows.y = points[:, 1], points[:, 0]
        entry.update({
            "localisations": rows, "_box_ids": identifiers,
            "origin": "napari_edit", "source": {"dataset": self.active_dataset, "channel": self.active_channel},
        })
        self.localisation_dict["bounding_boxes"] = entry
        with self.bbox_layer.events.data.blocker():
            self.bbox_layer.features = {"bounding_box_id": list(self.session.bounding_box_ids())}

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

    def closeEvent(self, event):
        self.stop_event.set()
        self.session.experiment.close()
        super().closeEvent(event)
