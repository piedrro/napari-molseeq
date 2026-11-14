__version__ = "2.0.0"

__all__ = (
    "QWidget",
)


def __getattr__(name):
    if name == "QWidget":
        from molseeq.ui.widget import QWidget

        return QWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

