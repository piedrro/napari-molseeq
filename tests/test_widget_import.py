def test_napari_entry_point_loads_with_locked_picasso():
    from qtpy.QtWidgets import QWidget

    from molseeq import QWidget as MolSeeqWidget

    assert issubclass(MolSeeqWidget, QWidget)


def test_widget_publishes_an_alex_import_to_the_napari_layer(
    qapp, tmp_path
):
    import numpy as np
    import tifffile
    from napari.components import ViewerModel

    from molseeq import QWidget
    from molseeq.session import ImportSettings, read_recording

    movie = np.arange(5 * 4 * 6, dtype=np.uint16).reshape(5, 4, 6)
    path = tmp_path / "alex.tif"
    tifffile.imwrite(path, movie, photometric="minisblack")
    viewer = ViewerModel()
    widget = QWidget(viewer)
    try:
        models = ["Translation", "Rigid", "Similarity", "Affine", "Homography"]
        for selector, default, choices in (
            (widget.gui.tform_compute_model, "Homography", models),
            (widget.gui.align_model, "Auto", ["Auto", *models]),
        ):
            assert selector.currentText() == default
            assert [selector.itemText(i) for i in range(selector.count())] == choices
        recordings = [read_recording(path, ImportSettings("alex"))]
        widget._publish_import(
            recordings, widget.session.revision, None, False
        )
        widget.update_active_image(dataset=path.name, channel="dd")
        np.testing.assert_array_equal(
            widget.image_layer.data, movie[::2, :, :3]
        )
        unrelated_pixels = np.full((4, 3), 123, dtype=np.uint16)
        unrelated = viewer.add_image(unrelated_pixels, name="Unrelated image")
        channel = widget.gui.picasso_channel
        dd = channel.findText("DD")
        if dd < 0:
            dd = channel.findText("dd")
        assert dd >= 0
        channel.setCurrentIndex(dd)
        assert widget._clear_detection_image()
        np.testing.assert_array_equal(widget.image_layer.data, np.zeros((4, 3)))
        np.testing.assert_array_equal(unrelated.data, unrelated_pixels)
        widget._picasso_wrapper_finished()
        np.testing.assert_array_equal(widget.image_layer.data, movie[::2, :, :3])
        widget.localisation_dict["bounding_boxes"] = {
            "localisations": np.rec.fromarrays(
                ([0], [1.0], [1.0]), names="frame,x,y"
            ),
            "box_size": 3,
            "fitted": False,
        }
        widget.draw_bounding_boxes()
        (identifier,) = widget.session.bounding_box_ids()
        widget.bbox_layer.data = np.array([[2.0, 1.0]])
        assert widget.session.bounding_box_ids() == (identifier,)
        assert (
            widget.session.dataset(path.name).bounding_boxes[identifier].y == 2
        )
        widget.bbox_layer.add([1.0, 2.0])
        first, added = widget.session.bounding_box_ids()
        assert first == identifier and added != identifier
        widget.bbox_layer.selected_data = {0}
        widget.bbox_layer.remove_selected()
        assert widget.session.bounding_box_ids() == (added,)
    finally:
        widget.close()
        viewer.layers.clear()
