
from typing import TYPE_CHECKING
from qtpy.QtWidgets import QPushButton
from qtpy.QtCore import QThreadPool
from qtpy.QtWidgets import (QWidget,QVBoxLayout, QFrame, QSizePolicy, QSlider, QComboBox,QLineEdit, QProgressBar, QLabel, QCheckBox, QGridLayout)
import numpy as np
import traceback
from multiprocessing import Manager
from functools import partial

from napari_pixseq.funcs.pixseq_utils_compute import _utils_compute
from napari_pixseq.funcs.pixseq_undrift_utils import _undrift_utils
from napari_pixseq.funcs.pixseq_picasso_detect import _picasso_detect_utils
from napari_pixseq.funcs.pixseq_loc_utils import _loc_utils
from napari_pixseq.funcs.pixseq_import_utils import _import_utils
from napari_pixseq.funcs.pixseq_events import _events_utils
from napari_pixseq.funcs.pixseq_export_images_utils import _export_images_utils
from napari_pixseq.funcs.pixseq_transform_utils import _tranform_utils
from napari_pixseq.funcs.pixseq_trace_compute_utils import _trace_compute_utils
from napari_pixseq.funcs.pixseq_plot_utils import _plot_utils, CustomPyQTGraphWidget
from napari_pixseq.funcs.pixseq_align_utils import _align_utils
from napari_pixseq.funcs.pixseq_export_traces_utils import _export_traces_utils
from napari_pixseq.funcs.pixseq_colocalize_utils import _utils_colocalize
from napari_pixseq.funcs.pixseq_temporal_filtering import _utils_temporal_filtering
from napari_pixseq.funcs.pixseq_cluster_utils import _cluster_utils

import napari



class PixSeqWidget(QWidget,
    _undrift_utils, _picasso_detect_utils,
    _import_utils, _events_utils, _export_images_utils,
    _tranform_utils, _trace_compute_utils, _plot_utils,
    _align_utils, _loc_utils, _export_traces_utils,
    _utils_colocalize, _utils_temporal_filtering, _utils_compute,
    _cluster_utils):

    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        from napari_pixseq.GUI.pixseq_ui import Ui_Frame

        #create UI
        self.setLayout(QVBoxLayout())
        self.form = Ui_Frame()
        self.pixseq_ui = QFrame()
        self.pixseq_ui.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.pixseq_ui.setMinimumSize(450, 500)

        self.form.setupUi(self.pixseq_ui)
        self.layout().addWidget(self.pixseq_ui)

        #create pyqt graph container
        self.graph_container = self.findChild(QWidget, "graph_container")
        self.graph_container.setLayout(QVBoxLayout())

        self.graph_canvas = CustomPyQTGraphWidget(self)
        self.graph_container.layout().addWidget(self.graph_canvas)

        # register controls
        self.pixseq_import_mode = self.findChild(QComboBox, 'pixseq_import_mode')
        self.pixseq_import_limt = self.findChild(QComboBox, 'pixseq_import_limt')
        self.pixseq_channel_layout = self.findChild(QComboBox, 'pixseq_channel_layout')
        self.pixseq_channel_layout_label = self.findChild(QLabel, 'pixseq_channel_layout_label')
        self.pixseq_alex_first_frame = self.findChild(QComboBox, 'pixseq_alex_first_frame')
        self.pixseq_alex_first_frame_label = self.findChild(QLabel, 'pixseq_alex_first_frame_label')
        self.pixseq_import = self.findChild(QPushButton, 'pixseq_import')
        self.pixseq_import_progressbar = self.findChild(QProgressBar, 'pixseq_import_progressbar')
        self.pixseq_append = self.findChild(QCheckBox, 'pixseq_append')
        self.pixseq_append_dataset = self.findChild(QComboBox, 'pixseq_append_dataset')
        self.pixseq_append_dataset_label = self.findChild(QLabel, 'pixseq_append_dataset_label')

        self.pixseq_old_dataset_name = self.findChild(QComboBox, 'pixseq_old_dataset_name')
        self.pixseq_new_dataset_name = self.findChild(QLineEdit, 'pixseq_new_dataset_name')
        self.pixseq_update_dataset_name = self.findChild(QPushButton, 'pixseq_update_dataset_name')

        self.delete_dataset_name = self.findChild(QComboBox, 'delete_dataset_name')
        self.pixseq_delete_dataset = self.findChild(QPushButton, 'pixseq_delete_dataset')

        self.update_labels_dataset = self.findChild(QComboBox, 'update_labels_dataset')
        self.sequence_label = self.findChild(QComboBox, 'sequence_label')
        self.gap_label = self.findChild(QComboBox, 'gap_label')
        self.pixseq_update_labels = self.findChild(QPushButton, 'pixseq_update_labels')

        self.pixseq_dataset_selector = self.findChild(QComboBox, 'pixseq_dataset_selector')
        self.pixseq_show_dd = self.findChild(QPushButton, 'pixseq_show_dd')
        self.pixseq_show_da = self.findChild(QPushButton, 'pixseq_show_da')
        self.pixseq_show_aa = self.findChild(QPushButton, 'pixseq_show_aa')
        self.pixseq_show_ad = self.findChild(QPushButton, 'pixseq_show_ad')

        self.import_alex_data = self.findChild(QPushButton, 'import_alex_data')
        self.channel_selector = self.findChild(QComboBox, 'channel_selector')

        self.import_picasso_type = self.findChild(QComboBox, "import_picasso_type")
        self.import_picasso_dataset = self.findChild(QComboBox, 'import_picasso_dataset')
        self.import_picasso_channel = self.findChild(QComboBox, 'import_picasso_channel')
        self.import_picasso = self.findChild(QPushButton, 'import_picasso')

        self.picasso_dataset = self.findChild(QComboBox, 'picasso_dataset')
        self.picasso_channel = self.findChild(QComboBox, 'picasso_channel')
        self.picasso_min_net_gradient = self.findChild(QLineEdit, 'picasso_min_net_gradient')
        self.picasso_roi_border_width = self.findChild(QLineEdit, 'picasso_roi_border_width')
        self.picasso_box_size = self.findChild(QComboBox, 'picasso_box_size')
        self.picasso_frame_mode = self.findChild(QComboBox, 'picasso_frame_mode')
        self.picasso_detect = self.findChild(QPushButton, 'picasso_detect')
        self.picasso_fit = self.findChild(QPushButton, 'picasso_fit')
        self.picasso_detectfit = self.findChild(QPushButton, 'picasso_detectfit')
        self.picasso_detect_mode = self.findChild(QComboBox, 'picasso_detect_mode')
        self.picasso_window_cropping = self.findChild(QCheckBox, 'picasso_window_cropping')
        self.picasso_remove_overlapping = self.findChild(QCheckBox, 'picasso_remove_overlapping')
        self.picasso_progressbar = self.findChild(QProgressBar, 'picasso_progressbar')

        self.picasso_vis_mode = self.findChild(QComboBox, 'picasso_vis_mode')
        self.picasso_vis_size = self.findChild(QComboBox, 'picasso_vis_size')
        self.picasso_vis_opacity = self.findChild(QComboBox, 'picasso_vis_opacity')
        self.picasso_vis_edge_width = self.findChild(QComboBox, 'picasso_vis_edge_width')

        self.picasso_vis_mode.currentIndexChanged.connect(partial(self.draw_fiducials, update_vis=True))
        self.picasso_vis_mode.currentIndexChanged.connect(partial(self.draw_bounding_boxes, update_vis=True))
        self.picasso_vis_size.currentIndexChanged.connect(partial(self.draw_fiducials, update_vis=True))
        self.picasso_vis_size.currentIndexChanged.connect(partial(self.draw_bounding_boxes, update_vis=True))
        self.picasso_vis_opacity.currentIndexChanged.connect(partial(self.draw_fiducials, update_vis=True))
        self.picasso_vis_opacity.currentIndexChanged.connect(partial(self.draw_bounding_boxes, update_vis=True))
        self.picasso_vis_edge_width.currentIndexChanged.connect(partial(self.draw_fiducials, update_vis=True))
        self.picasso_vis_edge_width.currentIndexChanged.connect(partial(self.draw_bounding_boxes, update_vis=True))

        self.cluster_localisations = self.findChild(QPushButton, 'cluster_localisations')
        self.cluster_mode = self.findChild(QComboBox, 'cluster_mode')
        self.cluster_channel = self.findChild(QComboBox, 'cluster_channel')
        self.cluster_dataset = self.findChild(QComboBox, 'cluster_dataset')
        self.cluster_eps = self.findChild(QLineEdit, "cluster_eps")
        self.dbscan_min_samples = self.findChild(QLineEdit, "dbscan_min_samples")

        self.undrift_dataset_selector = self.findChild(QComboBox, 'undrift_dataset_selector')
        self.undrift_channel_selector = self.findChild(QComboBox, 'undrift_channel_selector')
        self.picasso_undrift = self.findChild(QPushButton, 'picasso_undrift')
        self.undrift_progressbar = self.findChild(QProgressBar, 'undrift_progressbar')

        self.filtering_datasets = self.findChild(QComboBox, 'filtering_datasets')
        self.filtering_channels = self.findChild(QComboBox, 'filtering_channels')
        self.filtering_mode = self.findChild(QComboBox, 'filtering_mode')
        self.filtering_filter_size = self.findChild(QComboBox, 'filtering_filter_size')
        self.filtering_start = self.findChild(QPushButton, 'filtering_start')
        self.filtering_progressbar = self.findChild(QProgressBar, 'filtering_progressbar')

        self.align_reference_dataset = self.findChild(QComboBox, 'align_reference_dataset')
        self.align_reference_channel = self.findChild(QComboBox, 'align_reference_channel')
        self.pixseq_align_datasets = self.findChild(QPushButton, 'pixseq_align_datasets')
        self.align_progressbar = self.findChild(QProgressBar, 'align_progressbar')

        self.pixseq_import_tform = self.findChild(QPushButton, 'pixseq_import_tform')

        self.tform_compute_dataset = self.findChild(QComboBox, 'tform_compute_dataset')
        self.tform_compute_ref_channel = self.findChild(QComboBox, 'tform_compute_ref_channel')
        self.tform_compute_target_channel = self.findChild(QComboBox, 'tform_compute_target_channel')
        self.pixseq_compute_tform = self.findChild(QPushButton, 'pixseq_compute_tform')
        self.tform_apply_target = self.findChild(QComboBox, 'tform_apply_target')
        self.pixseq_apply_tform = self.findChild(QPushButton, 'pixseq_apply_tform')
        self.tform_apply_progressbar = self.findChild(QProgressBar, 'tform_apply_progressbar')
        self.save_tform = self.findChild(QCheckBox, 'save_tform')

        self.pixseq_link_localisations = self.findChild(QPushButton, 'pixseq_link_localisations')

        self.export_dataset = self.findChild(QComboBox, 'export_dataset')
        self.export_channel = self.findChild(QComboBox, 'export_channel')
        self.pixseq_export_data = self.findChild(QPushButton, 'pixseq_export_data')

        self.traces_export_mode = self.findChild(QComboBox, 'traces_export_mode')
        self.traces_export_dataset = self.findChild(QComboBox, 'traces_export_dataset')
        self.traces_export_channel = self.findChild(QComboBox, 'traces_export_channel')
        self.traces_export_metric = self.findChild(QComboBox, 'traces_export_metric')
        self.traces_export_background = self.findChild(QComboBox, 'traces_export_background')
        self.pixseq_export_traces = self.findChild(QPushButton, 'pixseq_export_traces')
        self.export_progressbar = self.findChild(QProgressBar, 'export_progressbar')

        self.locs_export_mode = self.findChild(QComboBox, 'locs_export_mode')
        self.locs_export_dataset = self.findChild(QComboBox, 'locs_export_dataset')
        self.locs_export_channel = self.findChild(QComboBox, 'locs_export_channel')
        self.pixseq_export_locs = self.findChild(QPushButton, 'pixseq_export_locs')

        self.traces_spot_size = self.findChild(QComboBox, "traces_spot_size")
        self.traces_spot_shape = self.findChild(QComboBox, "traces_spot_shape")
        self.traces_background_buffer = self.findChild(QComboBox, "traces_background_buffer")
        self.traces_background_width = self.findChild(QComboBox, "traces_background_width")
        self.compute_with_picasso = self.findChild(QCheckBox, "compute_with_picasso")
        self.compute_global_background = self.findChild(QCheckBox, "compute_global_background")
        self.traces_visualise_masks = self.findChild(QPushButton, 'traces_visualise_masks')
        self.traces_visualise_bg_masks = self.findChild(QPushButton, 'traces_visualise_bg_masks')
        self.traces_channel_selection_layout = self.findChild(QGridLayout, 'traces_channel_selection_layout')
        self.compute_traces = self.findChild(QPushButton, 'compute_traces')
        self.compute_traces_progressbar = self.findChild(QProgressBar, 'compute_traces_progressbar')

        self.plot_data = self.findChild(QComboBox, 'plot_data')
        self.plot_channel = self.findChild(QComboBox, 'plot_channel')
        self.plot_metric = self.findChild(QComboBox, 'plot_metric')
        self.plot_background_mode = self.findChild(QComboBox, 'plot_background_mode')
        self.split_plots = self.findChild(QCheckBox, 'split_plots')
        self.normalise_plots = self.findChild(QCheckBox, 'normalise_plots')
        self.focus_on_bbox = self.findChild(QCheckBox, 'focus_on_bbox')
        self.plot_compute_progress = self.findChild(QProgressBar, 'plot_compute_progress')
        self.plot_localisation_number = self.findChild(QSlider, 'plot_localisation_number')
        self.plot_localisation_number_label = self.findChild(QLabel, 'plot_localisation_number_label')

        self.colo_dataset = self.findChild(QComboBox, 'colo_dataset')
        self.colo_channel1 = self.findChild(QComboBox, 'colo_channel1')
        self.colo_channel2 = self.findChild(QComboBox, 'colo_channel2')
        self.colo_max_dist = self.findChild(QComboBox, 'colo_max_dist')
        self.colo_bboxes = self.findChild(QCheckBox, 'colo_bboxes')
        self.colo_fiducials = self.findChild(QCheckBox, 'colo_fiducials')
        self.pixseq_colocalize = self.findChild(QPushButton, 'pixseq_colocalize')

        self.dev_verbose = self.findChild(QCheckBox, 'dev_verbose')
        self.dev_verbose.stateChanged.connect(self.toggle_verbose)
        self.verbose = False

        self.pixseq_import.clicked.connect(self.pixseq_import_data)
        self.pixseq_import_mode.currentIndexChanged.connect(self.update_import_options)
        self.pixseq_update_dataset_name.clicked.connect(self.update_dataset_name)
        self.pixseq_delete_dataset.clicked.connect(self.delete_dataset)
        self.pixseq_update_labels.clicked.connect(self.update_nucleotide)

        self.import_picasso.clicked.connect(self.import_picaaso_localisations)

        self.picasso_detect.clicked.connect(partial(self.pixseq_picasso, detect = True, fit=False))
        self.picasso_fit.clicked.connect(partial(self.pixseq_picasso, detect = False, fit=True))
        self.picasso_detectfit.clicked.connect(partial(self.pixseq_picasso, detect=True, fit=True))
        self.cluster_localisations.clicked.connect(self.pixseq_cluster_localisations)
        self.dbscan_remove_overlapping = self.findChild(QCheckBox, "dbscan_remove_overlapping")


        self.pixseq_dataset_selector.currentIndexChanged.connect(self.update_channel_select_buttons)
        self.pixseq_dataset_selector.currentIndexChanged.connect(partial(self.update_active_image,
            dataset = self.pixseq_dataset_selector.currentText()))

        self.picasso_undrift.clicked.connect(self.undrift_images)

        self.pixseq_align_datasets.clicked.connect(self.align_datasets)
        self.align_reference_dataset.currentIndexChanged.connect(self.update_align_reference_channel)

        self.pixseq_import_tform.clicked.connect(self.import_transform_matrix)
        self.pixseq_compute_tform.clicked.connect(self.compute_transform_matrix)
        self.pixseq_apply_tform.clicked.connect(self.apply_transform_matrix)

        self.picasso_detect_mode.currentIndexChanged.connect(self.update_picasso_options)

        self.pixseq_export_data.clicked.connect(self.export_data)
        self.export_dataset.currentIndexChanged.connect(self.update_export_options)

        self.pixseq_export_locs.clicked.connect(self.initialise_export_locs)
        self.locs_export_mode.currentIndexChanged.connect(self.update_loc_export_options)
        self.locs_export_dataset.currentIndexChanged.connect(self.update_loc_export_options)

        self.pixseq_export_traces.clicked.connect(self.export_traces)
        self.traces_export_dataset.currentIndexChanged.connect(self.populate_export_combos)

        self.viewer.dims.events.current_step.connect(partial(self.draw_fiducials, update_vis = False))

        self.compute_traces.clicked.connect(self.pixseq_compute_traces)
        self.traces_visualise_masks.clicked.connect(self.visualise_spot_masks)
        self.traces_visualise_masks.clicked.connect(self.visualise_background_masks)

        self.plot_data.currentIndexChanged.connect(partial(self.update_plot_combos, combo="plot_data"))
        self.plot_channel.currentIndexChanged.connect(partial(self.update_plot_combos, combo="plot_channel"))

        self.plot_data.currentIndexChanged.connect(self.initialize_plot)
        self.plot_channel.currentIndexChanged.connect(self.initialize_plot)
        self.plot_metric.currentIndexChanged.connect(self.initialize_plot)
        self.split_plots.stateChanged.connect(self.initialize_plot)
        self.normalise_plots.stateChanged.connect(self.initialize_plot)
        self.plot_background_mode.currentIndexChanged.connect(self.initialize_plot)
        self.focus_on_bbox.stateChanged.connect(self.initialize_plot)

        self.pixseq_colocalize.clicked.connect(self.pixseq_colocalize_fiducials)

        self.plot_localisation_number.valueChanged.connect(lambda: self.update_slider_label("plot_localisation_number"))
        self.plot_localisation_number.valueChanged.connect(partial(self.plot_traces))

        self.filtering_start.clicked.connect(self.pixseq_temporal_filtering)
        self.filtering_datasets.currentIndexChanged.connect(self.update_filtering_channels)

        self.pixseq_append.stateChanged.connect(self.update_import_append_options)

        self.picasso_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="picasso_dataset", channel_selector="picasso_channel"))
        self.undrift_dataset_selector.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="undrift_dataset_selector", channel_selector="undrift_channel_selector"))
        self.cluster_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="cluster_dataset", channel_selector="cluster_channel"))
        self.tform_compute_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="tform_compute_dataset", channel_selector="tform_compute_ref_channel", channel_type="donor"))
        self.tform_compute_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="tform_compute_dataset", channel_selector="tform_compute_target_channel", channel_type="acceptor"))
        self.colo_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="colo_dataset", channel_selector="colo_channel1"))
        self.colo_dataset.currentIndexChanged.connect(partial(self.update_channel_selector, dataset_selector="colo_dataset", channel_selector="colo_channel2"))

        self.dataset_dict = {}
        self.localisation_dict = {"bounding_boxes": {}, "fiducials": {}}
        self.traces_dict = {}
        self.plot_dict = {}

        self.active_dataset = None
        self.active_channel = None

        self.threadpool = QThreadPool()

        manager = Manager()
        self.stop_event = manager.Event()

        self.worker = None
        self.multiprocessing_active = False

        self.transform_matrix = None

        self.update_import_options()
        self.update_import_append_options()

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

        self.viewer.bind_key('D', self.dev_function)

        self.viewer.bind_key('PageUp', self.named_partial(self.increment_active_dataset, key='Up'), overwrite=True)
        self.viewer.bind_key('PageDown', self.named_partial(self.increment_active_dataset, key='Down'), overwrite=True)

        self.viewer.bind_key('Q', self.stop_worker, overwrite=True)

        # todo dataset selection for multiple datasets?
        # todo picasso export localisations
        # todo delete dataset e.g. localisation dataset

    def toggle_verbose(self):

        if self.dev_verbose.isChecked():
            self.verbose = True
        else:
            self.verbose = False

    def dev_function(self, event):

        print("Dev function called")
        self.restore_shared_frames()
        self.update_ui()

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

            vis_mode = self.picasso_vis_mode.currentText()
            vis_size = float(self.picasso_vis_size.currentText())
            vis_opacity = float(self.picasso_vis_opacity.currentText())
            vis_edge_width = float(self.picasso_vis_edge_width.currentText())

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

                if "localisation_centres" in self.localisation_dict["bounding_boxes"].keys():

                    if self.verbose:
                        print("Drawing bounding_boxes")

                    loc_dict, n_locs, fitted = self.get_loc_dict(type = "bounding_boxes")

                    localisations = loc_dict["localisations"].copy()
                    localisation_centres = self.get_localisation_centres(localisations,mode="bounding_boxes")

                    # print(f"Drawing {len(localisation_centres)} bounding boxes")

                    vis_mode = self.picasso_vis_mode.currentText()
                    vis_size = float(self.picasso_vis_size.currentText())
                    vis_opacity = float(self.picasso_vis_opacity.currentText())
                    vis_edge_width = float(self.picasso_vis_edge_width.currentText())

                    if vis_mode.lower() == "square":
                        symbol = "square"
                    elif vis_mode.lower() == "disk":
                        symbol = "disc"
                    elif vis_mode.lower() == "x":
                        symbol = "cross"

                    if "bounding_boxes" not in layer_names:
                        self.bbox_layer = self.viewer.add_points(
                            localisation_centres,
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
                        self.viewer.layers["bounding_boxes"].data = localisation_centres

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


    def draw_fiducials(self, update_vis=False):

        remove_fiducials = True

        if hasattr(self, "localisation_dict") and hasattr(self, "active_channel"):

            if hasattr(self, "fiducial_layer"):
                show_fiducials = self.fiducial_layer.visible
            else:
                show_fiducials = True

            if show_fiducials:

                layer_names = [layer.name for layer in self.viewer.layers]

                active_frame = self.viewer.dims.current_step[0]

                dataset_name = self.pixseq_dataset_selector.currentText()
                image_channel = self.active_channel

                if image_channel != "" and dataset_name != "":

                    if image_channel.lower() in self.localisation_dict["fiducials"][dataset_name].keys():
                        localisation_dict = self.localisation_dict["fiducials"][dataset_name][image_channel.lower()].copy()

                        if "render_locs" in localisation_dict.keys():

                            render_locs = localisation_dict["render_locs"]

                            vis_mode = self.picasso_vis_mode.currentText()
                            vis_size = float(self.picasso_vis_size.currentText())
                            vis_opacity = float(self.picasso_vis_opacity.currentText())
                            vis_edge_width = float(self.picasso_vis_edge_width.currentText())

                            if vis_mode.lower() == "square":
                                symbol = "square"
                            elif vis_mode.lower() == "disk":
                                symbol = "disc"
                            elif vis_mode.lower() == "x":
                                symbol = "cross"

                            if active_frame in render_locs.keys():

                                remove_fiducials = False

                                if "fiducials" not in layer_names:

                                    if self.verbose:
                                        print("Drawing fiducials")

                                    self.fiducial_layer = self.viewer.add_points(
                                        render_locs[active_frame],
                                        ndim=2,
                                        edge_color="red",
                                        face_color=[0,0,0,0],
                                        opacity=vis_opacity,
                                        name="fiducials",
                                        symbol=symbol,
                                        size=vis_size,
                                        edge_width=vis_edge_width, )

                                    self.fiducial_layer.mouse_drag_callbacks.append(self._mouse_event)
                                    self.fiducial_layer.events.visible.connect(self.draw_fiducials)

                                    update_vis = True

                                else:

                                    if self.verbose:
                                        print("Updating fiducial data")

                                    self.fiducial_layer.data = render_locs[active_frame]
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



                if remove_fiducials:
                    if "fiducials" in layer_names:
                        self.viewer.layers["fiducials"].data = []

                for layer in layer_names:
                    self.viewer.layers[layer].refresh()


    def get_localisation_centres(self, locs, mode = "fiducials"):

        loc_centres = []

        try:

            for loc in locs:
                frame = int(loc.frame)
                if mode == "fiducials":
                    loc_centres.append([frame, loc.y, loc.x])
                else:
                    loc_centres.append([loc.y, loc.x])

        except:
            print(traceback.format_exc())

        return loc_centres

