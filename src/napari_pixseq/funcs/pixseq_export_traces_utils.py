import traceback
import os
from qtpy.QtWidgets import QFileDialog
from napari_pixseq.funcs.pixseq_utils_compute import Worker
from functools import partial
import numpy as np
import json
import pandas as pd
import originpro as op
import random
import string

class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


class _export_traces_utils:


    def get_export_traces_path(self, dialog=True):

        if self.traces_dict != {}:

            dataset_name = self.traces_export_dataset.currentText()
            export_channel = self.traces_export_channel.currentText()
            export_mode = self.traces_export_mode.currentText()

            if dataset_name == "All Datasets":
                dataset_name = list(self.traces_dict.keys())[0]
            if export_channel in ["All Channels", "FRET Efficiency", "ALEX Efficiency"]:
                export_channel = list(self.traces_dict[dataset_name].keys())[0]

            import_path = self.dataset_dict[dataset_name][export_channel.lower()]["path"]

            import_directory = os.path.dirname(import_path)
            file_name = os.path.basename(import_path)
            file_name, file_extension = os.path.splitext(file_name)

            if export_mode == "JSON Dataset":
                file_extension = ".json"
            elif export_mode == "Dat (.dat)":
                file_extension = ".dat"
            elif export_mode == "Text (.txt)":
                file_extension = ".txt"
            elif export_mode == "CSV (.csv)":
                file_extension = ".csv"
            elif export_mode == "Nero (.dat)":
                file_extension = ".dat"
            elif export_mode == "ebFRET SMD (.mat)":
                file_extension = ".mat"
            elif export_mode == "Excel":
                file_extension = ".xlsx"
            elif export_mode == "OriginLab":
                file_extension = ".opju"
            else:
                self.pixseq_notification(f"Export mode {export_mode} not recognized.")

            export_path = os.path.join(import_directory, file_name + file_extension)
            export_path = os.path.normpath(export_path)

            if dialog:
                export_path = QFileDialog.getSaveFileName(self, "Save File", export_path, "All Files (*)")[0]

            export_directory = os.path.dirname(export_path)

            return export_path, export_directory

    def export_traces_json(self, progress_callback=None, export_path=""):

        json_dict = self.populate_json_dict(progress_callback)

        with open(export_path, "w") as f:
            json.dump(json_dict, f, cls=npEncoder)

        self.traces_export_status = True

    def export_traces_dat(self, progress_callback=None, export_path=""):

        try:

            export_dict = self.populate_export_dict()

            export_dataset = np.stack(export_dict["data"], axis=0).T

            export_dataset = pd.DataFrame(export_dataset)

            export_dataset.columns = [export_dict["index"], export_dict["dataset"], export_dict["channel"]]

            export_dataset.to_csv(export_path, sep="\t", index=False, header=True)

        except:
            print(traceback.format_exc())
            pass

    def export_traces_txt(self, progress_callback=None, export_path=""):

        try:

            export_dict = self.populate_export_dict()

            export_dataset = np.stack(export_dict["data"], axis=0).T

            export_dataset = pd.DataFrame(export_dataset)

            export_dataset.columns = [export_dict["index"], export_dict["dataset"], export_dict["channel"]]

            export_dataset.to_csv(export_path, sep="\t", index=False, header=True)

        except:
            print(traceback.format_exc())
            pass

    def export_traces_csv(self, progress_callback=None, export_path=""):

        try:

            export_dict = self.populate_export_dict()

            export_dataset = np.stack(export_dict["data"], axis=0).T

            export_dataset = pd.DataFrame(export_dataset)

            export_dataset.columns = [export_dict["index"],
                                      export_dict["dataset"],
                                      export_dict["channel"]]

            export_dataset.to_csv(export_path, sep=",", index=False, header=True)

        except:
            print(traceback.format_exc())
            pass


    def export_traces_excel(self, progress_callback=None, export_path=""):

        try:
            export_dict = self.populate_export_dict()

            export_dataset = np.stack(export_dict["data"], axis=0).T

            export_dataset = pd.DataFrame(export_dataset)

            export_dataset.columns = [export_dict["index"], export_dict["dataset"], export_dict["channel"]]

            export_dataset.columns.names = ["index", "dataset", "channel"]

            with pd.ExcelWriter(export_path) as writer:
                export_dataset.to_excel(writer, sheet_name='Trace Data', index=True, startrow=1, startcol=1)

        except:
            print(traceback.format_exc())

    def export_traces_originlab(self, progress_callback=None, export_path=""):

        try:

            export_dict = self.populate_export_dict()

            export_data = np.stack(export_dict["data"], axis=0).T

            export_data = pd.DataFrame(export_data)

            export_data.columns = export_dict["channel"]

            if os.path.exists(export_path):
                os.remove(export_path)

            if op.oext:
                op.set_show(False)

            op.new()

            wks = op.new_sheet()
            wks.cols_axis('YY')
            wks.from_df(export_data, addindex=True)

            for i in range(len(export_dict["channel"])):
                index = export_dict["index"][i]
                dataset = export_dict["dataset"][i]

                wks.set_label(i, dataset, 'Dataset')
                wks.set_label(i, index, 'Index')

                if progress_callback is not None:
                    progress = int((i+1) / len(export_dict["channel"]) * 100)
                    progress_callback.emit(progress)

            export_path = os.path.normpath(export_path)
            op.save(export_path)

            if op.oext:
                op.exit()

        except:
            print(traceback.format_exc())


    def json_dict_report(self, json_dataset):

        try:

            if json_dataset != {}:

                json_report = {}
                dataset_traces = {}

                data_dict = json_dataset["data"]

                for dataset in data_dict.keys():

                    data = data_dict[dataset]

                    if dataset not in json_report.keys():
                        json_report[dataset] = {}

                    dataset_traces[dataset] = len(data)

                    for json_dict in data:

                        json_dict_keys = json_dict.keys()

                        for key in json_dict_keys:

                            if key not in json_report[dataset].keys():
                                json_report[dataset][key] = 0

                            json_report[dataset][key] += 1

                n_datasets = len(json_report.keys())
                unique_channels = list(set([key for dataset in json_report.keys() for key in json_report[dataset].keys()]))
                unique_n_traces = np.unique([value for dataset in json_report.keys() for value in json_report[dataset].values()])
                total_traces = sum([value for dataset in json_report.keys() for value in json_report[dataset].values()])

                # size of json_dataset
                json_dataset_size = len(json.dumps(json_dataset, indent=4, cls=npEncoder))
                json_dataset_size_mb = json_dataset_size / 1000000

                print(f"JSON Dataset report:")
                print(f" N datasets: {n_datasets}")
                print(f" Dataset traces: {list(dataset_traces.values())}")
                print(f" Unique channels: {unique_channels}")
                print(f" N traces: {unique_n_traces}")
                print(f" Total traces: {total_traces}")
                print(f" Size: {json_dataset_size_mb} MB")

        except:
            print(traceback.format_exc())

    def populate_json_dict(self, progress_callback=None):

        dataset_name = self.traces_export_dataset.currentText()
        channel_name = self.traces_export_channel.currentText()
        metric_name = self.traces_export_metric.currentText()
        background_mode = self.traces_export_background.currentText()

        metric_key = self.get_dict_key(self.metric_dict, metric_name)

        if background_mode not in ["None", None,""] and type(metric_key) == str:
            key_modifier = self.get_dict_key(self.background_dict, background_mode)
            background_metric_key = metric_key + key_modifier
        else:
            background_metric_key = None

        if dataset_name == "All Datasets":
            dataset_list = list(self.traces_dict.keys())
        else:
            dataset_list = [dataset_name]

        if channel_name == "All Channels":

            channel_list = [channel for dataset_dict in self.traces_dict.values() for channel in dataset_dict.keys()]
            channel_list = list(set(channel_list))
            channel_list = [channel for channel in channel_list if "efficiency" not in channel.lower()]
            iteration_channel = channel_list[0]
            if set(["dd", "da"]).issubset(channel_list):
                channel_list.append("alex_efficiency")
            if set(["donor", "acceptor"]).issubset(channel_list):
                channel_list.append("fret_efficiency")

        elif channel_name.lower() == "fret data":
            channel_list = ["donor", "acceptor"]
        elif channel_name.lower() == "alex data":
            channel_list = ["dd", "da", "ad", "aa"]
        else:
            channel_list = [channel_name.lower()]

        json_dict = {"metadata": {}, "data": {}}

        n_traces = 0
        for dataset in dataset_list:
            if dataset in self.traces_dict:
                for channel in channel_list:
                    if channel in self.traces_dict[dataset]:
                        n_traces += len(self.traces_dict[dataset][channel].copy())

        gap_label_dict = {}
        sequence_label_dict = {}

        loc_dict, n_locs, fitted = self.get_loc_dict(type="bounding_boxes")
        locs = loc_dict["localisations"]
        spot_locs_dict = {}

        print(channel_list,n_traces)

        for dataset in dataset_list:
            for channel_name, channel_dict in self.dataset_dict[dataset].items():
                if "gap_label" in channel_dict.keys():
                    gap_label_dict[dataset] = channel_dict["gap_label"]
                if "sequence_label" in channel_dict.keys():
                    sequence_label_dict[dataset] = channel_dict["sequence_label"]

                if dataset not in spot_locs_dict.keys():
                    spot_locs_dict[dataset] = {}

                traces_dict = self.traces_dict[dataset][channel_name]

                for trace_index, trace_dict in traces_dict.items():

                    if trace_index not in spot_locs_dict[dataset].keys():
                        spot_locs_dict[dataset][trace_index] = {}

                    spot_loc = locs[trace_index].copy().tolist()
                    spot_locs_dict[dataset][trace_index] = spot_loc

        iter = 0
        for dataset in dataset_list:

            if dataset not in json_dict["data"]:
                json_dict["data"][dataset] = []

            if dataset in gap_label_dict.keys():
                gap_label = gap_label_dict[dataset]
            else:
                gap_label = None

            if dataset in sequence_label_dict.keys():
                sequence_label = sequence_label_dict[dataset]
            else:
                sequence_label = None

            if channel_name == "All Channels" or "efficiency" in channel_name.lower():
                dataset_channels = self.traces_dict[dataset].keys()
                if set(["dd", "da"]).issubset(dataset_channels):
                    self.compute_alex_efficiency(dataset, metric_key, background_metric_key,
                        progress_callback, clip_data=False)
                elif set(["Donor", "Acceptor"]).issubset(dataset_channels):
                    self.compute_fret_efficiency(dataset, metric_key, background_metric_key,
                        progress_callback, clip_data=False)

            for channel in channel_list:

                if channel in self.traces_dict[dataset].keys():

                    channel_dict = self.traces_dict[dataset][channel].copy()

                    if json_dict["data"][dataset] == []:
                        json_dict["data"][dataset] = [{} for _ in range(n_traces)]

                    for trace_index, trace_dict in channel_dict.items():

                        if trace_index in spot_locs_dict[dataset].keys():
                            spot_loc = spot_locs_dict[dataset][trace_index]
                        else:
                            spot_loc = None

                        if channel.lower() in ["dd", "da", "ad", "aa"]:
                            channel_name = channel.upper()
                        elif channel.lower() == "alex_efficiency":
                            channel_name = "efficiency"
                        elif channel.lower() == "fret_efficiency":
                            channel_name = "efficiency"
                        else:
                            channel_name = channel.capitalize()

                        if channel_name not in json_dict["data"][dataset][trace_index]:
                            json_dict["data"][dataset][trace_index][channel_name] = []

                        if metric_key in trace_dict.keys():
                            data = trace_dict[metric_key].copy()

                            if "efficiency" not in channel and background_mode != "None":
                                background = np.array(trace_dict[background_metric_key].copy())
                                data = data - background

                            data = np.array(data).astype(float).tolist()

                            json_dict["data"][dataset][trace_index][channel_name] = data
                            json_dict["data"][dataset][trace_index]["gap_label"] = gap_label
                            json_dict["data"][dataset][trace_index]["sequence_label"] = sequence_label
                            json_dict["data"][dataset][trace_index]["picasso_loc"] = spot_loc

                        iter += 1

                        if progress_callback is not None:
                            progress_callback.emit(iter / n_traces * 100)

        return json_dict

    def populate_export_dict(self):

        export_dict = {}

        try:
            dataset_name = self.traces_export_dataset.currentText()
            channel_name = self.traces_export_channel.currentText()
            metric_name = self.traces_export_metric.currentText()
            background_mode = self.traces_export_background.currentText()

            metric_key = self.get_dict_key(self.metric_dict, metric_name)

            if background_mode not in ["None", None, ""] and type(metric_key) == str:
                key_modifier = self.get_dict_key(self.background_dict, background_mode)
                background_metric_key = metric_key + key_modifier
            else:
                background_metric_key = None

            if dataset_name == "All Datasets":
                dataset_list = list(self.traces_dict.keys())
            else:
                dataset_list = [dataset_name]

            if channel_name == "All Channels":
                channel_list = list(self.traces_dict[dataset_list[0]].keys())
            elif channel_name.lower() == "fret data":
                channel_list = ["donor", "acceptor"]
            elif channel_name.lower() == "alex data":
                channel_list = ["dd", "da", "ad", "aa"]
            elif channel_name.lower() == "alex efficiency":
                channel_list = ["alex_efficiency"]
            elif channel_name.lower() == "fret efficiency":
                channel_list = ["fret_efficiency"]
            else:
                channel_list = [channel_name.lower()]

            dataset_name_list = []
            channel_name_list = []
            index_list = []
            data_list = []

            n_traces = 0

            n_traces = len(self.traces_dict[dataset_list[0]][channel_list[0]].keys())

            for dataset in dataset_list:
                for trace_index in range(n_traces):
                    for channel in channel_list:

                        if trace_index in self.traces_dict[dataset][channel].keys():
                            trace_dict = self.traces_dict[dataset][channel][trace_index].copy()

                            if channel.lower() in ["dd", "da", "ad", "aa"]:
                                channel_name = channel.upper()
                            else:
                                channel_name = channel.capitalize()

                            if dataset not in export_dict.keys():
                                export_dict[dataset] = []

                            data = np.array(trace_dict[metric_key].copy())

                            if "efficiency" not in channel and background_mode != "None":
                                background = np.array(trace_dict[background_metric_key].copy())
                                data = data - background

                            data = data.astype(float).tolist()

                            dataset_name_list.append(dataset)
                            channel_name_list.append(channel_name)
                            index_list.append(trace_index)
                            data_list.append(data)

                export_dict["dataset"] = dataset_name_list
                export_dict["channel"] = channel_name_list
                export_dict["index"] = index_list
                export_dict["data"] = data_list

        except:
            print(traceback.format_exc())
            pass

        return export_dict

    def export_traces_finished(self, export_path):

        print("Traces exported to: {}".format(export_path))

        self.update_ui()

    def export_traces_error(self, error_message):

        self.update_ui()


    def export_traces(self):

        try:

            self.pixseq_export_traces.setEnabled(False)

            export_path, export_directory = self.get_export_traces_path(dialog=True)

            if export_path != "" and os.path.isdir(export_directory):

                self.update_ui(init=True)

                export_mode = self.traces_export_mode.currentText()

                if export_mode == "JSON Dataset":

                    self.worker = Worker(self.export_traces_json, export_path=export_path)
                    self.worker.signals.progress.connect(partial(self.pixseq_progress,
                        progress_bar=self.export_progressbar))
                    self.worker.signals.finished.connect(partial(self.export_traces_finished,
                        export_path=export_path))
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)

                elif export_mode == "Dat (.dat)":

                    self.worker = Worker(self.export_traces_dat, export_path=export_path)
                    self.worker.signals.progress.connect(partial(self.pixseq_progress,
                        progress_bar=self.export_progressbar))
                    self.worker.signals.finished.connect(partial(self.export_traces_finished,
                        export_path=export_path))
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)

                elif export_mode == "Text (.txt)":

                    self.worker = Worker(self.export_traces_txt, export_path=export_path)
                    self.worker.signals.progress.connect(partial(self.pixseq_progress,
                        progress_bar=self.export_progressbar))
                    self.worker.signals.finished.connect(partial(self.export_traces_finished,
                        export_path=export_path))
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)

                elif export_mode == "CSV (.csv)":

                    self.worker = Worker(self.export_traces_csv, export_path=export_path)
                    self.worker.signals.progress.connect(partial(self.pixseq_progress,
                        progress_bar=self.export_progressbar))
                    self.worker.signals.finished.connect(partial(self.export_traces_finished,
                        export_path=export_path))
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)

                elif export_mode == "Excel":

                    self.worker = Worker(self.export_traces_excel, export_path=export_path)
                    self.worker.signals.progress.connect(partial(self.pixseq_progress,
                        progress_bar=self.export_progressbar))
                    self.worker.signals.finished.connect(partial(self.export_traces_finished,
                        export_path=export_path))
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)


                elif export_mode == "OriginLab":

                    self.worker = Worker(self.export_traces_originlab, export_path=export_path)
                    self.worker.signals.progress.connect(partial(self.pixseq_progress,
                        progress_bar=self.export_progressbar))
                    self.worker.signals.finished.connect(partial(self.export_traces_finished,
                        export_path=export_path))
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)

                elif export_mode == "Nero (.dat)":

                    self.worker = Worker(self.export_traces_nero, export_path=export_path)
                    self.worker.signals.progress.connect(partial(self.pixseq_progress,
                        progress_bar=self.export_progressbar))
                    self.worker.signals.finished.connect(partial(self.export_traces_finished,
                        export_path=export_path))
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)

                elif export_mode == "ebFRET SMD (.mat)":

                    self.worker = Worker(self.export_traces_ebfret_smd, export_path=export_path)
                    self.worker.signals.progress.connect(partial(self.pixseq_progress,
                        progress_bar=self.export_progressbar))
                    self.worker.signals.finished.connect(partial(self.export_traces_finished,
                        export_path=export_path))
                    self.worker.signals.error.connect(self.update_ui)
                    self.threadpool.start(self.worker)

                else:
                    self.update_ui()

            else:
                self.update_ui()

        except:
            print(traceback.format_exc())
            self.update_ui()

    def export_traces_nero(self, export_path, progress_callback=None):

        try:
            export_dict = self.populate_export_dict()

            export_data = np.stack(export_dict["data"], axis=0).T

            export_indices = np.array(export_dict["index"]).astype(int)
            export_indices += 1

            export_data = pd.DataFrame(export_data)
            export_data.columns = export_indices

            export_data.to_csv(export_path, index=False, sep=" ")

        except:
            print(traceback.format_exc())

    def populate_smd_dict(self):

        try:

            dataset_name = self.traces_export_dataset.currentText()
            channel_name = self.traces_export_channel.currentText()
            metric_name = self.traces_export_metric.currentText()
            background_mode = self.traces_export_background.currentText()

            metric_key = self.get_dict_key(self.metric_dict, metric_name)

            if background_mode not in ["None", None, ""] and type(metric_key) == str:
                key_modifier = self.get_dict_key(self.background_dict, background_mode)
                background_metric_key = metric_key + key_modifier
            else:
                background_metric_key = None

            if dataset_name == "All Datasets":
                dataset_list = list(self.traces_dict.keys())
            else:
                dataset_list = [dataset_name]

            if channel_name == "All Channels":
                channel_list = list(self.traces_dict[dataset_list[0]].keys())
            elif channel_name.lower() == "fret data":
                channel_list = ["donor", "acceptor"]
            elif channel_name.lower() == "alex data":
                channel_list = ["dd", "da", "ad", "aa"]
            elif channel_name.lower() == "alex efficiency":
                channel_list = ["alex_efficiency"]
            elif channel_name.lower() == "fret efficiency":
                channel_list = ["fret_efficiency"]
            else:
                channel_list = [channel_name.lower()]

            smd_dict = {"attr": {"data_package": "TraceAnalyser"},
                        "columns": channel_list,
                        "data": {"attr": [], "id": [], "index": [], "values": []},
                        "id": ''.join(random.choices(string.ascii_lowercase + string.digits, k=32)),
                        "type": "TraceAnalyser",
                        }

            n_traces = len(self.traces_dict[dataset_list[0]][channel_list[0]].keys())



            for dataset in dataset_list:

                export_channel = list(self.traces_dict[dataset].keys())[0]
                file_path = self.dataset_dict[dataset_name][export_channel.lower()]["path"]

                for trace_index in range(n_traces):

                    smd_values = []

                    for channel in channel_list:

                        if trace_index in self.traces_dict[dataset][channel].keys():
                            trace_dict = self.traces_dict[dataset][channel][trace_index].copy()

                            data = np.array(trace_dict[metric_key].copy())

                            if "efficiency" not in channel and background_mode != "None":
                                background = np.array(trace_dict[background_metric_key].copy())
                                data = data - background

                            smd_values.append(data)

                    smd_index = np.expand_dims(np.arange(1, len(smd_values[0])+1), -1).astype(float).tolist()
                    smd_values = np.stack(smd_values, axis=1).tolist()

                    lowerbound = np.min(smd_values)

                    smd_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))
                    smd_attr = {
                        "file": os.path.basename(file_path),
                        "layer": channel,
                        "localisation_number": trace_index,
                        "lowerbound": lowerbound,
                        "group": "group 1",
                        "restart":0,
                        "crop_min": 0,
                        "crop_max": len(data),
                    }

                    smd_dict["data"]["attr"].append(smd_attr)
                    smd_dict["data"]["id"].append(smd_id)
                    smd_dict["data"]["index"].append(smd_index)
                    smd_dict["data"]["values"].append(smd_values)

        except:
            print(traceback.format_exc())
            smd_dict = {}

        return smd_dict

    def export_traces_ebfret_smd(self, export_path, progress_callback=None):

            try:

                smd_dict = self.populate_smd_dict()

                import mat4py
                mat4py.savemat(export_path, smd_dict)

            except:
                print(traceback.format_exc())

    def populate_export_combos(self):

        try:
            export_channel_list = []

            export_mode = self.traces_export_mode.currentText()
            export_dataset = self.traces_export_dataset.currentText()

            if export_dataset != "":

                if export_dataset == "All Datasets":

                    for dataset_name, dataset_dict in self.dataset_dict.items():
                        for channel_name in dataset_dict.keys():

                            if channel_name.lower() in ["donor", "acceptor"]:
                                channel_name = channel_name.capitalize()
                            else:
                                channel_name = channel_name.upper()

                            export_channel_list.append(channel_name)
                            export_channel_list = list(set(export_channel_list))

                else:

                    if export_dataset in self.dataset_dict.keys():
                        for channel_name in self.dataset_dict[export_dataset].keys():

                            if channel_name.lower() in ["donor", "acceptor"]:
                                channel_name = channel_name.capitalize()
                            else:
                                channel_name = channel_name.upper()

                            export_channel_list.append(channel_name)

                if set(["Donor", "Acceptor"]).issubset(set(export_channel_list)):
                    export_channel_list.insert(0, "FRET Data")
                    export_channel_list.insert(0, "FRET Efficiency")
                if set(["DD","DA","AA","AD"]).issubset(set(export_channel_list)):
                    export_channel_list.insert(0, "ALEX Data")
                    export_channel_list.insert(0, "ALEX Efficiency")

                export_channel_list.insert(0, "All Channels")

                if export_mode == "Nero (.dat)":
                    multi_channel_list = ["All Channels", "FRET Data", "ALEX Data"]
                    export_channel_list = [chan for chan in export_channel_list if chan not in multi_channel_list]

                self.traces_export_channel.blockSignals(True)
                self.traces_export_channel.clear()
                self.traces_export_channel.addItems(export_channel_list)
                self.traces_export_channel.blockSignals(True)

                # if hasattr(self, "traces_dict"):
                #
                #     dataset_name = self.traces_export_dataset.currentText()
                #     channel_name = self.traces_export_channel.currentText()
                #
                #     if dataset_name in self.traces_dict.keys():
                #
                #         if dataset_name == "All Datasets":
                #             dataset_names = list(self.traces_dict.keys())
                #             dataset_name = dataset_names[0]
                #
                #         if channel_name.lower() == "fret":
                #             channel_name = "donor"
                #         if channel_name.lower() == "alex":
                #             channel_name = "dd"
                #         if channel_name.lower() == "all channels":
                #             channel_names = list(self.traces_dict[dataset_name].keys())
                #             channel_names = [chan for chan in channel_names if "efficiency" not in chan.lower()]
                #             channel_names = [chan for chan in channel_names if chan.lower() not in ["fret", "alex"]]
                #             channel_name = channel_names[0]
                #
                #         if dataset_name in self.traces_dict.keys():
                #             if channel_name.lower() in self.traces_dict[dataset_name].keys():
                #
                #                 traces_channel_dict = self.traces_dict[dataset_name][channel_name.lower()]
                #                 metric_names = traces_channel_dict[0].keys()
                #
                #                 # self.traces_export_metric.blockSignals(True)
                #                 #
                #                 # self.traces_export_metric.clear()
                #                 #
                #                 # for metric in metric_names:
                #                 #     if metric in self.metric_dict.keys():
                #                 #         self.traces_export_metric.addItem(self.metric_dict[metric])
                #                 #
                #                 # self.traces_export_metric.blockSignals(False)

        except:
            print(traceback.format_exc())
            pass
