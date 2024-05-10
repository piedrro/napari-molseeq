import traceback
import numpy as np
import cv2
from napari_pixseq.funcs.pixseq_utils_compute import Worker


class _utils_colocalize:



    def filter_locs_by_matches(self, locs, coords, matches):

        try:

            locs = locs.copy()
            coords = coords.copy()
            matches = matches.copy()

            coords = np.float32([coords[m.queryIdx] for m in matches]).reshape(-1, 2)

            filtered_locs = []
            filtered_render_locs = {}

            for loc in locs:
                coord = [loc.x, loc.y]
                frame = loc.frame
                if coord in coords.tolist():
                    filtered_locs.append(loc)

                    if frame not in filtered_render_locs.keys():
                        filtered_render_locs[frame] = []

                    filtered_render_locs[frame].append([loc.y, loc.x])

            filtered_locs = np.rec.fromrecords(filtered_locs, dtype=locs.dtype)
            filtered_loc_centers = self.get_localisation_centres(filtered_locs)


        except:
            print(traceback.format_exc())
            filtered_locs = None
            filtered_render_locs = None
            filtered_loc_centers = None
            pass

        return filtered_locs, filtered_render_locs, filtered_loc_centers





    def _pixseq_colocalize_fiducials(self, progress_callback=None):

        try:

            dataset = self.gui.colo_dataset.currentText()
            channel1 = self.gui.colo_channel1.currentText()
            channel2 = self.gui.colo_channel2.currentText()
            max_dist = float(self.gui.colo_max_dist.currentText())

            ch1_loc_dict, ch1_n_locs, _ = self.get_loc_dict(dataset, channel1.lower())
            ch2_loc_dict, ch2_n_locs, _ = self.get_loc_dict(dataset, channel2.lower())

            ch1_locs = ch1_loc_dict["localisations"].copy()
            ch2_locs = ch2_loc_dict["localisations"].copy()

            ch1_coords = [[loc.x, loc.y] for loc in ch1_locs]
            ch2_coords = [[loc.x, loc.y] for loc in ch2_locs]

            ch1_coords = np.array(ch1_coords).astype(np.float32)
            ch2_coords = np.array(ch2_coords).astype(np.float32)

            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

            matches = bf.match(ch1_coords, ch2_coords)
            matches = [m for m in matches if m.distance < max_dist]

            ch1_coords = np.float32([ch1_coords[m.queryIdx] for m in matches]).reshape(-1, 2)
            ch2_coords = np.float32([ch2_coords[m.trainIdx] for m in matches]).reshape(-1, 2)

            filtered_ch1_locs = []
            filter_ch1_render_locs = {}

            for loc in ch1_locs:
                coord = [loc.x, loc.y]
                frame = loc.frame
                if coord in ch1_coords.tolist():
                    filtered_ch1_locs.append(loc)

                    if frame not in filter_ch1_render_locs.keys():
                        filter_ch1_render_locs[frame] = []

                    filter_ch1_render_locs[frame].append([loc.y, loc.x])

            filtered_ch1_locs = np.rec.fromrecords(filtered_ch1_locs, dtype=ch1_locs.dtype)
            filtered_ch1_loc_centers = self.get_localisation_centres(filtered_ch1_locs)

            filtered_ch2_locs = []
            filter_ch2_render_locs = {}

            for loc in ch2_locs:
                coord = [loc.x, loc.y]
                frame = loc.frame
                if coord in ch2_coords.tolist():
                    filtered_ch2_locs.append(loc)

                    if frame not in filter_ch2_render_locs.keys():
                        filter_ch2_render_locs[frame] = []

                    filter_ch2_render_locs[frame].append([loc.y, loc.x])

            filtered_ch2_locs = np.rec.fromrecords(filtered_ch2_locs, dtype=ch2_locs.dtype)
            filtered_ch2_loc_centers = self.get_localisation_centres(filtered_ch2_locs)

            colo_locs = []
            colo_render_locs = {}

            for loc1, loc2 in zip(filtered_ch1_locs, filtered_ch2_locs):
                locX = (loc1.x + loc2.x) / 2
                locY = (loc1.y + loc2.y) / 2
                coord = [locX, locY]
                frame = loc1.frame

                colo_loc = loc1.copy()
                colo_loc.x = locX
                colo_loc.y = locY

                colo_locs.append(colo_loc)

                if frame not in colo_render_locs.keys():
                    colo_render_locs[frame] = []

                colo_render_locs[frame].append([locY, locX])

            colo_locs = np.rec.fromrecords(colo_locs, dtype=filtered_ch1_locs.dtype)
            colo_loc_centers = self.get_localisation_centres(colo_locs)

            result_dict = {"localisations": colo_locs,
                           "localisation_centres": colo_loc_centers,
                           "render_locs": colo_render_locs}

            self.pixseq_notification(f"Found {len(colo_locs)} colocalisations between {channel1} and {channel2}")

        except:
            print(traceback.format_exc())
            ch1_result_dict = None
            ch2_result_dict = None
            result_dict = None
            pass

        return result_dict

    def _pixseq_colocalize_fiducials_result(self, colo_locs):

        try:

            if colo_locs is not None:

                dataset = self.gui.colo_dataset.currentText()
                channel1 = self.gui.colo_channel1.currentText()
                channel2 = self.gui.colo_channel2.currentText()

                if self.gui.colo_fiducials.isChecked():

                    self.localisation_dict["fiducials"][dataset][channel1.lower()]["localisations"] = colo_locs["localisations"]
                    self.localisation_dict["fiducials"][dataset][channel1.lower()]["localisation_centres"] = colo_locs["localisation_centres"]
                    self.localisation_dict["fiducials"][dataset][channel1.lower()]["render_locs"] = colo_locs["render_locs"]

                    self.localisation_dict["fiducials"][dataset][channel2.lower()]["localisations"] = colo_locs["localisations"]
                    self.localisation_dict["fiducials"][dataset][channel2.lower()]["localisation_centres"] = colo_locs["localisation_centres"]
                    self.localisation_dict["fiducials"][dataset][channel2.lower()]["render_locs"] = colo_locs["render_locs"]

                    self.draw_fiducials(update_vis=True)

                if self.gui.colo_bboxes.isChecked():

                    self.localisation_dict["bounding_boxes"]["localisations"] = colo_locs["localisations"]
                    self.localisation_dict["bounding_boxes"]["localisation_centres"] = colo_locs["localisation_centres"]

                    self.draw_bounding_boxes()

        except:
            print(traceback.format_exc())
            pass

    def _pixseq_colocalize_fiducials_finished(self):

        self.update_ui()


    def pixseq_colocalize_fiducials(self):

        try:

            dataset = self.gui.colo_dataset.currentText()
            channel1 = self.gui.colo_channel1.currentText()
            channel2 = self.gui.colo_channel2.currentText()

            ch1_loc_dict, ch1_n_locs, _ = self.get_loc_dict(dataset, channel1.lower())
            ch2_loc_dict, ch2_n_locs, _ = self.get_loc_dict(dataset, channel2.lower())

            if channel1 == channel2:
                self.pixseq_notification("Channels must be different for colocalisation")

            elif ch1_n_locs == 0 or ch2_n_locs == 0:
                self.pixseq_notification("No localisations found in one or both channels.")

            else:
                self.update_ui(init=True)

                self.worker = Worker(self._pixseq_colocalize_fiducials)
                self.worker.signals.result.connect(self._pixseq_colocalize_fiducials_result)
                self.worker.signals.finished.connect(self._pixseq_colocalize_fiducials_finished)
                self.worker.signals.error.connect(self.update_ui)
                self.threadpool.start(self.worker)

        except:
            print(traceback.format_exc())
            pass


