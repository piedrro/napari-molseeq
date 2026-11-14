"""Configure the CPU bundle in both the launcher and spawned workers."""

import faulthandler
import os
import sys
from pathlib import Path


os.environ["QT_API"] = "pyqt6"
os.environ["NUMBA_DISABLE_CUDA"] = "1"

base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "MolSeeq"
logs = base / "logs"
try:
    logs.mkdir(parents=True, exist_ok=True)
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(logs / "console.log", "a", encoding="utf-8", buffering=1))
    _crash_stream = open(logs / "native-crash.log", "a", encoding="utf-8")
    faulthandler.enable(file=_crash_stream, all_threads=True)
except OSError:
    # Leave startup error reporting to the launcher when the profile is unwritable.
    pass
