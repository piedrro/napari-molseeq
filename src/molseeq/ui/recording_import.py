"""Qt import controls backed by the project session."""

import os
import traceback
from functools import partial

from qtpy.QtWidgets import QFileDialog

from molseeq.session import ImportSettings, read_recording
from molseeq.ui.worker import Worker


def _read_recordings(paths, settings, cancel_event, progress_callback):
    recordings = []
    for index, path in enumerate(paths, 1):
        if cancel_event.is_set():
            return ()
        recordings.append(read_recording(path, settings))
        progress_callback.emit(index * 100 // len(paths))
    return recordings


class RecordingImportMixin:

    def format_import_path(self, path):
        return os.path.normpath(path)

    def populate_dataset_combos(self):

        try:

            if self.verbose:
                print("Populating all dataset combos.")

            dataset_names = list(self.dataset_dict.keys())

            self.gui.molseeq_dataset_selector.blockSignals(True)
            self.gui.molseeq_dataset_selector.clear()
            self.gui.molseeq_dataset_selector.addItems(dataset_names)
            self.gui.molseeq_dataset_selector.blockSignals(False)

            self.gui.cluster_dataset.blockSignals(True)
            self.gui.cluster_dataset.clear()
            self.gui.cluster_dataset.addItems(dataset_names)
            self.gui.cluster_dataset.blockSignals(False)

            self.gui.tform_compute_dataset.blockSignals(True)
            self.gui.tform_compute_dataset.clear()
            self.gui.tform_compute_dataset.addItems(dataset_names)
            self.gui.tform_compute_dataset.blockSignals(False)

            self.gui.molseeq_old_dataset_name.blockSignals(True)
            self.gui.molseeq_old_dataset_name.clear()
            self.gui.molseeq_old_dataset_name.addItems(dataset_names)
            self.gui.molseeq_old_dataset_name.blockSignals(False)

            self.gui.align_reference_dataset.blockSignals(True)
            self.gui.align_reference_dataset.clear()
            self.gui.align_reference_dataset.addItems(dataset_names)
            self.gui.align_reference_dataset.blockSignals(False)

            self.gui.colo_dataset.blockSignals(True)
            self.gui.colo_dataset.clear()
            self.gui.colo_dataset.addItems(dataset_names)
            self.gui.colo_dataset.blockSignals(False)

            self.gui.molseeq_append_dataset.blockSignals(True)
            self.gui.molseeq_append_dataset.clear()
            self.gui.molseeq_append_dataset.addItems(dataset_names)
            self.gui.molseeq_append_dataset.blockSignals(False)

            self.gui.delete_dataset_name.blockSignals(True)
            self.gui.delete_dataset_name.clear()
            self.gui.delete_dataset_name.addItems(dataset_names)
            self.gui.delete_dataset_name.blockSignals(False)

            self.gui.update_labels_dataset.blockSignals(True)
            self.gui.update_labels_dataset.clear()
            self.gui.update_labels_dataset.addItems(dataset_names)
            self.gui.update_labels_dataset.blockSignals(False)

            self.gui.import_picasso_dataset.blockSignals(True)
            self.gui.import_picasso_dataset.clear()
            self.gui.import_picasso_dataset.addItems(dataset_names)
            self.gui.import_picasso_dataset.blockSignals(False)

            self.gui.picasso_filter_dataset.blockSignals(True)
            self.gui.picasso_filter_dataset.clear()
            self.gui.picasso_filter_dataset.addItems(dataset_names)
            self.gui.picasso_filter_dataset.blockSignals(False)

            self.gui.tracking_dataset.blockSignals(True)
            self.gui.tracking_dataset.clear()
            self.gui.tracking_dataset.addItems(dataset_names)
            self.gui.tracking_dataset.blockSignals(False)

            if len(dataset_names) > 1:
                dataset_names.insert(0, "All Datasets")

            self.gui.traces_export_dataset.blockSignals(True)
            self.gui.traces_export_dataset.clear()
            self.gui.traces_export_dataset.addItems(dataset_names)
            self.gui.traces_export_dataset.blockSignals(False)

            self.gui.filtering_datasets.blockSignals(True)
            self.gui.filtering_datasets.clear()
            self.gui.filtering_datasets.addItems(dataset_names)
            self.gui.filtering_datasets.blockSignals(False)

            self.gui.picasso_dataset.blockSignals(True)
            self.gui.picasso_dataset.clear()
            self.gui.picasso_dataset.addItems(dataset_names)
            self.gui.picasso_dataset.blockSignals(False)

            self.gui.undrift_dataset_selector.blockSignals(True)
            self.gui.undrift_dataset_selector.clear()
            self.gui.undrift_dataset_selector.addItems(dataset_names)
            self.gui.undrift_dataset_selector.blockSignals(False)

            self.gui.export_dataset.blockSignals(True)
            self.gui.export_dataset.clear()
            self.gui.export_dataset.addItems(dataset_names)
            self.gui.export_dataset.blockSignals(False)

            self.gui.locs_export_dataset.blockSignals(True)
            self.gui.locs_export_dataset.clear()
            self.gui.locs_export_dataset.addItems(dataset_names)
            self.gui.locs_export_dataset.blockSignals(False)

            self.gui.simple_plot_dataset.blockSignals(True)
            self.gui.simple_plot_dataset.clear()
            self.gui.simple_plot_dataset.addItems(dataset_names)
            self.gui.simple_plot_dataset.blockSignals(False)

            self.gui.picasso_render_dataset.blockSignals(True)
            self.gui.picasso_render_dataset.clear()
            self.gui.picasso_render_dataset.addItems(dataset_names)
            self.gui.picasso_render_dataset.blockSignals(False)

        except:
            print(traceback.format_exc())

    def _molseeq_import_data_finished(self):

        if self.verbose:
            print("Finished importing data, executing post import functions")

        self.populate_dataset_combos()

        self.update_channel_select_buttons()
        self.populate_channel_selectors()
        self.update_active_image()
        self.update_export_options()
        self.populate_export_combos()
        self.update_filtering_channels()
        self.update_loc_export_options()

        self.update_align_reference_channel()

        self.update_ui()

    def _publish_import(self, recordings, revision, append_to, concatenate):
        if not recordings or self.stop_event.is_set() or self.session.revision != revision:
            return
        try:
            self.session.import_recordings(
                recordings, append_to=append_to, concatenate=concatenate
            )
        except ValueError as error:
            self.molseeq_notification(str(error))
            return
        self._molseeq_import_data_finished()

    def molseeq_import_data(self):
        append_to = None
        if self.gui.molseeq_append.isChecked():
            append_to = self.gui.molseeq_append_dataset.currentText()
            if append_to not in self.session.images:
                self.molseeq_notification("Select a dataset to append to.")
                return

        desktop = os.path.expanduser("~/Desktop")
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open file", desktop, "Image files (*.tif *.tiff *.fits)"
        )
        if not paths:
            return

        limit = self.gui.molseeq_import_limt.currentText()
        settings = ImportSettings(
            self.gui.molseeq_import_mode.currentText(),
            self.gui.molseeq_channel_layout.currentText(),
            self.gui.molseeq_alex_first_frame.currentText(),
            None if limit == "None" else int(limit),
            self.gui.molseeq_pixel_size.value() or None,
        )
        revision = self.session.revision
        concatenate = self.gui.molseeq_concatenate.isChecked()
        self.update_ui(init=True)
        self.worker = Worker(_read_recordings, paths, settings, self.stop_event)
        self.worker.signals.progress.connect(self.gui.molseeq_import_progressbar.setValue)
        self.worker.signals.result.connect(
            partial(
                self._publish_import,
                revision=revision,
                append_to=append_to,
                concatenate=concatenate,
            )
        )
        self.worker.signals.error.connect(
            lambda error: self.molseeq_notification(str(error[1]))
        )
        self.worker.signals.finished.connect(self.update_ui)
        self.threadpool.start(self.worker)
