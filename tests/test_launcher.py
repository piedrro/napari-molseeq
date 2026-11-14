from molseeq import __version__


def test_desktop_viewer_has_versioned_molseeq_dock(qapp):
    from molseeq.launcher import create_viewer

    viewer = create_viewer(show=False)
    try:
        title = f"MolSeeq v{__version__}"
        widget = viewer.window.dock_widgets[title]
        assert widget.parentWidget().windowTitle() == title
        assert widget.windowTitle() == title
    finally:
        viewer.close()


def test_plugin_dock_keeps_version_when_floating_and_redocking(qtbot):
    import napari
    from qtpy.QtCore import Qt

    viewer = napari.Viewer(show=False)
    try:
        window = viewer.window._qt_window
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        dock, _ = viewer.window.add_plugin_dock_widget("napari-molseeq", "napari-molseeq")
        viewer.show()
        title = f"MolSeeq v{__version__}"
        qtbot.waitUntil(lambda: dock.windowTitle() == title)
        for floating in (True, False):
            dock.setFloating(floating)
            assert dock.windowTitle() == title
            assert dock.titleBarWidget().title.text() == title
    finally:
        viewer.close()
