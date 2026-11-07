def test_napari_entry_point_loads_with_locked_picasso():
    from qtpy.QtWidgets import QWidget

    from molseeq import QWidget as MolSeeqWidget

    assert issubclass(MolSeeqWidget, QWidget)


def test_widget_publishes_an_alex_import_to_the_napari_layer(
    qapp, tmp_path, monkeypatch
):
    from threading import Event
    from types import SimpleNamespace

    import numpy as np
    import tifffile
    from napari.components import ViewerModel

    from molseeq import molseeq_widget
    from molseeq.session import ImportSettings, read_recording

    monkeypatch.setattr(
        molseeq_widget, "Manager", lambda: SimpleNamespace(Event=Event)
    )
    movie = np.arange(5 * 4 * 6, dtype=np.uint16).reshape(5, 4, 6)
    path = tmp_path / "alex.tif"
    tifffile.imwrite(path, movie, photometric="minisblack")
    viewer = ViewerModel()
    widget = molseeq_widget.QWidget(viewer)
    try:
        recordings = [read_recording(path, ImportSettings("alex"))]
        widget._publish_import(
            recordings, widget.session.revision, None, False
        )
        widget.update_active_image(dataset=path.name, channel="dd")
        np.testing.assert_array_equal(
            widget.image_layer.data, movie[::2, :, :3]
        )
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
