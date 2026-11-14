from concurrent.futures import CancelledError
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from molcore import ArraySource

from molseeq.ui.image_export import write_image_jobs


def test_cancelled_image_batch_keeps_previous_files(tmp_path):
    first, second = tmp_path / "first.tif", tmp_path / "second.tif"
    first.write_bytes(b"previous export")
    source = ArraySource(np.ones((2, 4, 4), dtype=np.uint16))
    stop = Event()
    jobs = [
        (first, source.shape, source.dtype, iter(source)),
        (second, source.shape, source.dtype, iter(source)),
    ]

    with pytest.raises(CancelledError):
        write_image_jobs(
            jobs, stop, SimpleNamespace(emit=lambda _: stop.set())
        )

    assert first.read_bytes() == b"previous export"
    assert not second.exists()
