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
    finally:
        widget.close()
        viewer.layers.clear()
