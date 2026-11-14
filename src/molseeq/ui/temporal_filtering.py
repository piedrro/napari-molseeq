import traceback
from functools import partial

from molseeq.ui.worker import Worker


def _filter_sources(sources, size, method, stop_event, progress_callback):
    from molcore.temporal import rolling_filter

    results = {}
    for index, (key, source) in enumerate(sources.items()):
        filtered = rolling_filter(
            source, size, method=method, cancel_event=stop_event,
            progress=lambda update: progress_callback.emit(
                int((index + update.percent / 100) * 100 / len(sources))
            ),
        )
        results[key] = filtered.array
    return results

class TemporalFilteringMixin:


    def update_filtering_channels(self):

        try:

            if self.dataset_dict != {}:

                filtering_datasets = self.gui.filtering_datasets.currentText()

                if filtering_datasets == "All Datasets":
                    dataset_names = list(self.dataset_dict.keys())
                    channel_names = list(self.dataset_dict[dataset_names[0]].keys())
                else:
                    dataset_names = [filtering_datasets]
                    channel_names = list(self.dataset_dict[dataset_names[0]].keys())

                for channel_index, channel_name in enumerate(channel_names):

                    if channel_name.lower() in ["dd","da","ad","aa"]:
                        channel_name = channel_name.upper()
                    else:
                        channel_name = channel_name.capitalize()

                    channel_names[channel_index] = channel_name

                if len(channel_names) > 0:
                    channel_names.insert(0, "All Channels")

                self.gui.filtering_channels.clear()
                self.gui.filtering_channels.addItems(channel_names)

        except:
            print(traceback.format_exc())
            pass

    def molseeq_temporal_filtering(self, viewer=None):

        try:
            self.molseeq_notification("Starting temporal filtering...")

            self.gui.filtering_start.setEnabled(False)

            self.update_ui(init=True)

            self._start_core_filter()

        except:
            self.update_ui()
            print(traceback.format_exc())
            pass

    def _start_core_filter(self):
        name = self.gui.filtering_datasets.currentText()
        selection = self.gui.filtering_channels.currentText()
        sources = {
            (dataset, channel): self.session.working_source(dataset, channel)
            for dataset in self.session.images
            if name in ("All Datasets", dataset)
            for channel in self.session.images[dataset]
            if selection == "All Channels" or selection.lower() == channel
            if len(self.session.channel(dataset, channel))
        }
        size = int(self.gui.filtering_filter_size.currentText())
        method = "mean" if "mean" in self.gui.filtering_mode.currentText().lower() else "median"
        self.worker = Worker(_filter_sources, sources, size, method, self.stop_event)
        self.worker.signals.progress.connect(self.gui.filtering_progressbar.setValue)
        self.worker.signals.result.connect(partial(
            self._publish_filtered_images, revision=self.session.revision,
            operation={"method": "temporal_filter", "radius": size, "reduction": method},
        ))
        self.worker.signals.error.connect(lambda error: self.molseeq_notification(str(error[1])))
        self.worker.signals.finished.connect(self._molseeq_temporal_filtering_finished)
        self.threadpool.start(self.worker)

    def _publish_filtered_images(self, images, revision, operation):
        if not self.stop_event.is_set() and self.session.revision == revision:
            self.session.replace_working_images(images, operation=operation)

    def _molseeq_temporal_filtering_finished(self):

        try:
            self.image_layer.data = self.dataset_dict[self.active_dataset][self.active_channel]["data"]

            self.update_ui()

        except:
            print(traceback.format_exc())
            pass
