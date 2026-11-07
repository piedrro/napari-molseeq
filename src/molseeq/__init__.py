__version__ = "1.0.8"

__all__ = (
    "QWidget",
)


def __getattr__(name):
    if name == "QWidget":
        from molseeq.molseeq_widget import QWidget

        return QWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

