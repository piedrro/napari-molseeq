import traceback
from napari.utils.notifications import show_info



class _segmentation_events:

    def get_seglayer(self):

        layer_names = [layer.name for layer in self.viewer.layers]

        if "Segmentations" not in layer_names:
            self.segLayer = self.viewer.add_shapes(name="Segmentations", shape_type="polygon",
                opacity=0.5, face_color="red", edge_color="black", edge_width=1)

            self.segLayer.mouse_drag_callbacks.append(self.segevent)

            #keybinds for segmentation



        return self.segLayer

    def segmentation_modify_mode(self, viewer=None, mode=None):

        try:

            self.segLayer = self.get_seglayer()

            print(f"Segmentation mode: {mode}")

            if mode == "add":

                self.viewer.layers.selection.select_only(self.segLayer)

                self.interface_mode = "segment"
                self.segmentation_mode = "add"
                self.segLayer.mode = "add_polygon_lasso"
                show_info("Add (click/drag to add)")

            if mode == "extend":
                self.viewer.layers.selection.select_only(self.segLayer)

                self.interface_mode = "segment"
                self.segmentation_mode = "extend"
                show_info("Extend (click/drag to extend)")

            if mode == "join":
                self.viewer.layers.selection.select_only(self.segLayer)

                self.interface_mode = "segment"
                self.segmentation_mode = "join"
                show_info("Join (click/drag to join)")

            if mode == "split":

                self.viewer.layers.selection.select_only(self.segLayer)

                self.interface_mode = "segment"
                self.segmentation_mode = "split"
                show_info("Split (click/drag to split)")

            if mode == "delete":
                self.viewer.layers.selection.select_only(self.segLayer)

                self.interface_mode = "segment"
                self.segmentation_mode = "delete"

                self.segLayer.mode = "select"

                show_info("Delete (click/drag to delete)")

        except:
            print(traceback.format_exc())
            pass

    def get_selected_shape_index(self, event=None):

        try:

            position = event.position
            coord = [position[-2], position[-1]]

            shape_index = self.segLayer.get_value(coord)[0]

            return shape_index

        except:
            print(traceback.format_exc())
            pass


    def segevent(self, viewer=None, event=None):

        try:


            # print(self.segmentation_mode)
            #
            # if self.segmentation_mode == "extend":
            #
            #     selected_shape = self.get_selected_shape_index(event)
            #
            #     if selected_shape is not None:
            #         print(f"Selected shape: {selected_shape}")
            #     else:
            #         #delete last shape
            #         print("Delete last shape")

            if self.segmentation_mode == "delete":

                selected_shape = self.get_selected_shape_index(event)

                if selected_shape is not None:
                    shapes = self.segLayer.data.copy()
                    shapes.pop(selected_shape)
                    self.segLayer.data = shapes

            self.segLayer.selected = False

        except:
            print(traceback.format_exc())
            pass
