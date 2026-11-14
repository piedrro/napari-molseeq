import traceback
from functools import partial

import numpy as np
from molcore import StaleResultError
from molcore.staging import state_token, transact

from molseeq.model_analysis import excluded_localisation_ids
from molseeq.ui.worker import Worker


class LocalisationFilteringMixin:

    def molseeq_filter_localisations(self, viewer=None):
        localisation_type = self.gui.picasso_filter_type.currentText()
        dataset = self.gui.picasso_filter_dataset.currentText()
        channel = self.gui.picasso_filter_channel.currentText().lower()
        criterion = self.gui.filter_criterion.currentText()
        minimum = self.gui.filter_min.value()
        maximum = self.gui.filter_max.value()
        if not dataset or not channel or not criterion:
            return
        try:
            if localisation_type == "Localisations":
                self._start_localisation_filter(dataset, channel, criterion, minimum, maximum)
            else:
                entry, count, _ = self.get_loc_dict(dataset, channel, type="bounding_boxes")
                if count:
                    rows = entry["localisations"]
                    keep = (rows[criterion] > minimum) & (rows[criterion] < maximum)
                    if not np.all(keep):
                        entry["localisations"] = rows[keep]
                        self.localisation_dict["bounding_boxes"] = entry
                        self.draw_bounding_boxes(update_vis=True)
                    self.update_criterion_ranges()
                    print(f"Filtered {count - np.count_nonzero(keep)} {localisation_type}")
        except Exception:
            self.update_ui()
            print(traceback.format_exc())

    def _start_localisation_filter(self, dataset, channel, criterion, minimum, maximum):
        model = self.session.dataset(dataset)
        snapshot = model.snapshot()
        parts = tuple(self.session.channel_parts(dataset, channel))
        expected = state_token(model)
        revision = self.session.revision
        self.update_ui(init=True)
        worker = Worker(
            excluded_localisation_ids, snapshot, parts, channel, criterion, minimum, maximum,
            cancel_event=self.stop_event,
        )
        worker.signals.result.connect(partial(
            self._publish_filtered_localisations, dataset=dataset,
            expected=expected, revision=revision,
        ))
        worker.signals.error.connect(
            lambda error: self.molseeq_notification(str(error[1]))
            if not self.stop_event.is_set() else None
        )
        worker.signals.finished.connect(self._finish_localisation_filter)
        self.threadpool.start(worker)

    def _publish_filtered_localisations(self, excluded, dataset, expected, revision):
        if self.stop_event.is_set() or self.session.revision != revision:
            return
        if len(excluded):
            try:
                transact(
                    self.session.dataset(dataset),
                    lambda staged: staged.localizations.delete(excluded),
                    expected=expected,
                )
            except StaleResultError as error:
                self.molseeq_notification(str(error))
                return
            self.session.revision += 1
            self.draw_localisations(update_vis=True)
        print(f"Filtered {len(excluded)} Localisations")

    def _finish_localisation_filter(self):
        self.update_ui()
        self.update_criterion_ranges()

    def update_filter_dataset(self, viewer=None):

        if self.gui.picasso_filter_type.currentText() == "Localisations":
            self.gui.picasso_filter_dataset.setEnabled(True)
            self.gui.picasso_filter_dataset.show()
            self.gui.picasso_filter_dataset_label.show()
        else:
            self.gui.picasso_filter_dataset.setEnabled(False)
            self.gui.picasso_filter_dataset.hide()
            self.gui.picasso_filter_dataset_label.hide()

        self.update_filter_criterion()
        self.update_criterion_ranges()

    def update_filter_criterion(self, viewer=None):


        try:

            columns = []

            dataset = self.gui.picasso_filter_dataset.currentText()
            channel = self.gui.picasso_filter_channel.currentText()
            localisation_type = self.gui.picasso_filter_type.currentText()
            selector = self.gui.filter_criterion

            if dataset != "" and channel != "":

                if localisation_type == "Localisations":
                    loc_dict, n_locs, fitted = self.get_loc_dict(dataset, channel.lower(), type="localisations")
                else:
                    loc_dict, n_locs, fitted = self.get_loc_dict(dataset, channel.lower(), type="bounding_boxes")

                if n_locs > 0:

                    locs = loc_dict["localisations"].copy()

                    columns = list(locs.dtype.names)

                    columns = [col for col in columns if col not in ["dataset","channel"]]


            selector.clear()

            if len(columns) > 0:
                selector.addItems(columns)

        except:
            print(traceback.format_exc())


    def update_criterion_ranges(self, viewer=None, plot=True):

        try:

            self.filter_graph_canvas.clear()

            dataset = self.gui.picasso_filter_dataset.currentText()
            channel = self.gui.picasso_filter_channel.currentText()
            localisation_type = self.gui.picasso_filter_type.currentText()
            criterion = self.gui.filter_criterion.currentText()

            if localisation_type == "Localisations":
                loc_dict, n_locs, fitted = self.get_loc_dict(dataset, channel.lower(), type="localisations")
            else:
                loc_dict, n_locs, fitted = self.get_loc_dict(dataset, channel.lower(), type="bounding_boxes")

            if n_locs > 0:

                locs = loc_dict["localisations"].copy()

                columns = list(locs.dtype.names)

                if criterion in columns:

                    values = locs[criterion]

                    if values.dtype in [np.float32, np.float64, np.int32, np.int64]:

                        if plot:
                            self.plot_filter_graph(criterion, values)

                        min_value = np.min(values)
                        max_value = np.max(values)

                        self.gui.filter_min.setMinimum(min_value)
                        self.gui.filter_min.setMaximum(max_value)

                        self.gui.filter_max.setMinimum(min_value)
                        self.gui.filter_max.setMaximum(max_value)

                        self.gui.filter_min.setValue(min_value)
                        self.gui.filter_max.setValue(max_value)

        except:
            print(traceback.format_exc())

    def plot_filter_graph(self, criterion = "", values = None):

        try:
            self.filter_graph_canvas.clear()

            if values is not None:

                values = values[~np.isnan(values)]

                if len(values) > 0:
                    ax = self.filter_graph_canvas.addPlot()

                    # Create histogram
                    y, x = np.histogram(values, bins=100)

                    ax.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 75))
                    ax.setLabel('bottom', f"{criterion} values")
                    ax.setLabel('left', 'Frequency')

        except:
            print(traceback.format_exc())



