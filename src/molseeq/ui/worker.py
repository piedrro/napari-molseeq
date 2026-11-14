"""Run background jobs and deliver their results through Qt signals."""

import traceback

from qtpy.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    def __init__(self, callback, *args, **kwargs):
        super().__init__()
        self.callback = callback
        self.args = args
        self.signals = WorkerSignals()
        self.kwargs = {**kwargs, "progress_callback": self.signals.progress}

    @Slot()
    def run(self):
        try:
            result = self.callback(*self.args, **self.kwargs)
        except Exception as error:
            details = traceback.format_exc()
            traceback.print_exc()
            self.signals.error.emit((type(error), error, details))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
