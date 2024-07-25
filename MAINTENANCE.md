# Development

## Running and editing live running code from iPython console

To launch iPython, run the following command in your terminal with the napari-molseeq environment activated:

    ipython

Use autoreload to reload modules when they are changed, allowing you to modify the running code without restarting the kernel.

    %load_ext autoreload
    %autoreload 2

This will launch and instance of the viewer and the plugin widget.

    from napari import Viewer
    viewer = Viewer()
    from molseeq import QWidget
    viewer.window.add_dock_widget(QWidget(viewer))

After modifying the code, you can reload the PixSeq widget module with:

    QWidget

## Editing the UI

The UI is created using Qt Designer. To edit the UI, open the `gui.ui` file (src/GUI/gui.ui) in Qt Designer. 

The Qt Designer can be installed with the following command:

    conda install pyqt5-tools

The Qt Designer can be launched with the following command:

    qt5-tools designer

Once the UI is edited, the `gui.py` file can be generated with the following command:

    pyuic5 src/molseeq/GUI/gui.ui -o src/molseeq/GUI/gui.py

    OR
    
    cd src/molseeq/GUI
    pyuic5 gui.ui -o gui.py

## Deployment

PyPI version control is handled from the molseeq `__init__.py` file. To update the version, manually update the version number in the `src/molseeq/__init__.py` file before building and uploading the package to PyPi.
The package is built and uploaded to PyPI using the following commands:

    python -m build    
    python -m twine upload dist/*  

