"""Launch the napari-based MolSeeq desktop application."""

import logging
import multiprocessing
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def create_viewer(*, show=True):
    import napari

    from molseeq import QWidget

    viewer = napari.Viewer(show=False)
    try:
        widget = QWidget(viewer)
        viewer.window.add_dock_widget(widget, name=widget.windowTitle(), area="right")
        if show:
            viewer.show()
    except Exception:
        viewer.close()
        raise
    return viewer


def _configure_logging(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for key, subdirectory in (
        ("NUMBA_CACHE_DIR", "numba"),
        ("MPLCONFIGDIR", "matplotlib"),
    ):
        cache = Path(os.environ.setdefault(key, str(directory.parent / "cache" / subdirectory)))
        cache.mkdir(parents=True, exist_ok=True)
    os.environ["QT_API"] = "pyqt6"
    handler = RotatingFileHandler(
        directory / "molseeq.log", maxBytes=5_000_000, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _smoke_test(viewer, destination):
    import json
    from concurrent.futures import ProcessPoolExecutor
    from tempfile import TemporaryDirectory

    import numpy as np
    import tifffile
    from molcore.staging import experiment_state_token

    from molseeq import __version__
    from molseeq.session import ImportSettings, read_recording
    from molseeq.trace_views import compute_fixed_traces, publish_traces

    title = f"MolSeeq v{__version__}"
    widget = viewer.window.dock_widgets[title]
    if widget.parentWidget().windowTitle() != title:
        raise RuntimeError("The MolSeeq dock did not display the package version")
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        worker = pool.submit(os.getpid).result(timeout=60)
    if worker == os.getpid():
        raise RuntimeError("Smoke test did not spawn a separate worker")
    with TemporaryDirectory() as folder:
        movie = Path(folder) / "smoke.tif"
        tifffile.imwrite(movie, np.ones((2, 9, 9), dtype=np.uint16), photometric="minisblack")
        try:
            widget.session.import_recordings([read_recording(movie, ImportSettings("single channel"))])
            sites = np.rec.fromarrays(([4.0], [4.0]), names="x,y")
            traces = compute_fixed_traces(
                widget.session.image_snapshot(), sites, (3, "square", 0, 1, False, False),
                expected=experiment_state_token(widget.session.experiment),
            )
            publish_traces(widget.session, traces)
            sequences = widget.session.dataset(movie.name).photon_sequences.query()
            if len(sequences) != 1 or list(sequences[0].frames) != [0, 1]:
                raise RuntimeError("MolCore photometry returned no traces")
        finally:
            widget.session.experiment.close()
    Path(destination).write_text(
        json.dumps({"ok": True, "version": __version__, "worker": worker, "dock": title}, indent=2),
        encoding="utf-8",
    )


def main(argv=None):
    multiprocessing.freeze_support()
    import argparse

    parser = argparse.ArgumentParser(description="MolSeeq desktop application")
    parser.add_argument("--smoke-test", metavar="REPORT.json",
                        help="verify desktop startup and write a JSON report")
    args = parser.parse_args(argv)
    directory = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "MolSeeq" / "logs"
    handler = None
    viewer = None
    try:
        handler = _configure_logging(directory)
        viewer = create_viewer(show=args.smoke_test is None)
        if args.smoke_test:
            _smoke_test(viewer, args.smoke_test)
        else:
            import napari

            napari.run()
        return 0
    except Exception:
        logging.getLogger(__name__).exception("MolSeeq desktop launch failed")
        if not args.smoke_test and sys.platform == "win32":
            import ctypes

            details = f"See the logs in:\n{directory}" if handler else f"Could not open:\n{directory}"
            ctypes.windll.user32.MessageBoxW(None, details, "MolSeeq could not start", 0x10)
        return 1
    finally:
        if viewer is not None:
            viewer.close()
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
