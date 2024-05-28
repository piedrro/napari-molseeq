## Development

### Running from ipython terminal

Use autoreload to reload modules when they are changed, allowing you to modify the running code without restarting the kernel.

    %load_ext autoreload
    %autoreload 2

This will launch and instnace of the viewer and the plugin widget.

    from napari import Viewer
    viewer = Viewer()
    from napari_pixseq import PixSeqWidget
    viewer.window.add_dock_widget(PixSeqWidget(viewer))

After modifying the code, you can reload the PixSeq widget module with:

    PixSeqWidget

## Editing the UI

The UI is created using Qt Designer. To edit the UI, open the `pixseq.ui` file (src/GUI/pixseq_ui.ui) in Qt Designer. 

The Qt Designer can be installed with the following command:

    conda install pyqt5-tools

The Qt Designer can be launched with the following command:

    qt5-tools designer

Once the UI is edited, the `pixseq_ui.py` file can be generated with the following command:

    pyuic5 src/napari_pixseq/GUI/pixseq_ui.ui -o src/napari_pixseq/GUI/pixseq_ui.py


## Deployment

Version control is handled using Git tags. 
Increment the Git tag version number and push the tag to the repository.
While working in the root directory of the repository, run the following commands:

    git tag -a vX.X.X -m "PixSeq vX.X.X"
    python -m build
    python -m twine upload dist/*  
