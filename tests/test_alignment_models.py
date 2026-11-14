from concurrent.futures import CancelledError
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest


def test_detached_channel_transform_handles_one_frame_and_cancellation():
    from molcore import ArraySource

    from molseeq.ui.channel_transforms import compute_channel_transform

    image = np.zeros((1, 9, 11), dtype=np.uint16)
    image[0, 4, 4] = 100
    matrix = np.eye(3)
    matrix[0, 2] = 2
    sources = {"round": {"acceptor": ArraySource(image)}}
    stop = Event()
    result = compute_channel_transform(sources, matrix, "translation", stop)
    warped = result["round"].images["acceptor"]
    assert warped.shape == image.shape
    assert warped[0, 4, 6] == 100
    assert image[0, 4, 4] == 100
    stop.set()
    with pytest.raises(CancelledError):
        compute_channel_transform(sources, matrix, "translation", stop)


def selector(text):
    return SimpleNamespace(currentText=lambda: text)


def fiducials(points):
    return [SimpleNamespace(x=x, y=y) for x, y in points]


def example(model):
    source = np.array([[4., 4.], [13., 5.], [5., 14.], [16., 16.]])
    rotation = np.array([[np.cos(.03), -np.sin(.03)], [np.sin(.03), np.cos(.03)]])
    matrix = np.eye(3)
    matrix[:2, :2] = (1.04 if model == "Similarity" else 1.) * rotation
    matrix[:2, 2] = [1., -1.]
    return source, source @ matrix[:2, :2].T + matrix[:2, 2], matrix


@pytest.mark.parametrize("model", ["Rigid", "Similarity"])
def test_channel_model_is_used(model):
    from molseeq.ui.channel_transforms import (
        ChannelTransformsMixin,
        compute_channel_transform,
    )

    source, target, expected = example(model)
    tool = ChannelTransformsMixin()
    tool.dataset_dict = {"round": {}}
    tool.localisation_dict = {"localisations": {"round": {
        "acceptor": {"localisations": fiducials(source)},
        "donor": {"localisations": fiducials(target)},
    }}}
    tool.gui = SimpleNamespace(
        tform_compute_dataset=selector("round"),
        tform_compute_ref_channel=selector("donor"),
        tform_compute_target_channel=selector("acceptor"),
        tform_compute_model=selector(model),
        save_tform=SimpleNamespace(isChecked=lambda: False),
    )
    tool.molseeq_notification = lambda message: None
    tool.compute_transform_matrix()
    np.testing.assert_allclose(tool.transform_matrix, expected, atol=1e-5)
    assert tool.transform_model == model.lower()

    image = np.zeros((2, 20, 21), dtype=np.uint16)
    image[:, 4, 4] = 100
    progress = []
    corrections = compute_channel_transform(
        {"round": {"acceptor": image}}, tool.transform_matrix, tool.transform_model,
        Event(), progress_callback=SimpleNamespace(emit=progress.append),
    )
    result = corrections["round"]
    assert result.operation["model"] == model.lower()
    assert progress[-1] == 100
    warped = result.images["acceptor"]
    assert warped.shape == image.shape
    assert np.unravel_index(warped[0].argmax(), warped[0].shape) == (3, 5)

    tool.localisation_dict["localisations"]["round"]["acceptor"]["localisations"] = fiducials(source[:1])
    tool.compute_transform_matrix()
    assert tool.transform_matrix is None
    assert tool.transform_model is None


@pytest.mark.parametrize("model", ["Rigid", "Similarity", "Auto"])
def test_dataset_model_and_failed_fit(model):
    from molseeq.ui.dataset_alignment import compute_dataset_alignment

    source, target, expected = example(model)
    sources = {
        name: {"donor": np.ones((2, 20, 21), dtype=np.uint16)}
        for name in ("first", "second")
    }
    progress = SimpleNamespace(emit=lambda value: None)
    stop = Event()
    pairs = {"ref": fiducials(target), "first": fiducials(source), "second": fiducials(source[:1])}
    with pytest.raises(ValueError, match="second"):
        compute_dataset_alignment(sources, pairs, "ref", "rigid", stop, progress)

    pairs["second"] = fiducials(source)
    corrections = compute_dataset_alignment(sources, pairs, "ref", model.lower(), stop, progress)
    assert len(corrections) == 2
    for result in corrections.values():
        assert result.images["donor"].shape == (2, 20, 21)
        assert result.operation["model"] == ("homography" if model == "Auto" else model.lower())
        np.testing.assert_allclose(result.operation["matrix"], expected, atol=1e-5)
