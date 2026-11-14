from threading import Event
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile
from molcore.staging import state_token
from picasso.render import render

from molseeq.model_analysis import (
    excluded_localisation_ids,
    render_localisations,
)
from molseeq.session import ImportSettings, ProjectSession, read_recording
from molseeq.ui.events import ViewerEventsMixin


@pytest.fixture
def sequence(tmp_path):
    session = ProjectSession()
    recordings = []
    for index in range(2):
        path = tmp_path / f"movie{index}.tif"
        tifffile.imwrite(
            path, np.zeros((5, 8, 16), dtype=np.uint16), photometric="minisblack"
        )
        recordings.append(read_recording(path, ImportSettings("alex")))
    session.import_recordings(recordings, concatenate=True)
    name = recordings[0].name
    yield session, name
    session.experiment.close()


def test_filter_uses_strict_numeric_bounds_and_keeps_model_ids(sequence):
    session, name = sequence
    channel = "ad"
    session.localisations[name][channel]["localisations"] = np.rec.fromarrays(
        ([0, 1, 2, 3], [1.0, 2.0, 2.0, 3.0], [3.0] * 4),
        names="frame,x,y",
    )
    model = session.dataset(name)
    original = model.localizations.query(channel=channel)
    excluded = excluded_localisation_ids(
        model.snapshot(), tuple(session.channel_parts(name, channel)),
        channel, "x", 1.0, 3.0,
    )
    np.testing.assert_array_equal(excluded, original["id"][[0, 3]])
    expected = state_token(model)
    widget = _filter_context(session)
    widget._publish_filtered_localisations(excluded, name, expected, session.revision)
    np.testing.assert_array_equal(
        model.localizations.query(channel=channel)["id"], original["id"][[1, 2]]
    )


def test_filter_frame_uses_concatenated_channel_clock_and_rejects_stale_result(sequence):
    session, name = sequence
    channel = "ad"
    session.localisations[name][channel]["localisations"] = np.rec.fromarrays(
        ([0, 1, 2, 3], [2.0] * 4, [3.0] * 4), names="frame,x,y"
    )
    model = session.dataset(name)
    original = model.localizations.query(channel=channel)
    np.testing.assert_array_equal(original["frame"], [1, 3, 1, 3])
    expected = state_token(model)
    excluded = excluded_localisation_ids(
        model.snapshot(), tuple(session.channel_parts(name, channel)),
        channel, "frame", 0.0, 3.0,
    )
    np.testing.assert_array_equal(excluded, original["id"][[0, 3]])
    widget = _filter_context(session)
    widget.stop_event.set()
    widget._publish_filtered_localisations(excluded, name, expected, session.revision)
    widget.stop_event.clear()
    model.update_metadata({"edited": True})
    widget._publish_filtered_localisations(excluded, name, expected, session.revision)
    assert widget.notifications
    np.testing.assert_array_equal(model.localizations.query(channel=channel)["id"], original["id"])


def _filter_context(session):
    from molseeq.ui.localisation_filtering import LocalisationFilteringMixin

    widget = LocalisationFilteringMixin()
    widget.session = session
    widget.stop_event = Event()
    widget.draw_localisations = lambda **kwargs: None
    widget.notifications = []
    widget.molseeq_notification = widget.notifications.append
    return widget


def test_render_matches_pooled_picasso_precision_across_a_sequence(sequence):
    session, name = sequence
    channel = "ad"
    rows = np.rec.fromarrays(
        ([0, 2], [2.0, 5.0], [2.0, 5.0], [0.2, 0.9], [0.3, 1.1]),
        names="frame,x,y,lpx,lpy",
    )
    session.localisations[name][channel]["localisations"] = rows
    shape = session.channel(name, channel).shape
    image, pixel_size, oversampling = render_localisations(
        session.localisations[name][channel]["localisations"], shape,
        blur_method="convolve", min_blur_width=0.0, oversampling=4,
    )
    _, expected = render(
        pd.DataFrame(rows), {"Pixelsize": 1.0},
        disp_px_size=0.25, viewport=((0, 0), shape[-2:]),
        blur_method="convolve",
    )
    assert (pixel_size, oversampling) == (1, 4)
    np.testing.assert_allclose(image, expected)


def test_auto_contrast_uses_first_ten_frames_and_retains_cached_settings():
    widget = ViewerEventsMixin()
    widget.contrast_dict = {}
    widget.image_layer = SimpleNamespace()
    movie = np.arange(12 * 4 * 5, dtype=np.uint16).reshape(12, 4, 5)
    expected = np.percentile(movie[:10], [0.1, 99.99])
    limits = widget.image_layer_auto_contrast(movie, "movie", "data")
    np.testing.assert_allclose(limits, expected)
    np.testing.assert_allclose(widget.image_layer.contrast_limits, expected)
    assert widget.image_layer.gamma > 0
    widget.contrast_dict = {"movie": {"data": {"contrast_limits": (0, 2), "gamma": 0.4}}}
    assert widget.image_layer_auto_contrast(movie, "movie", "data") == (0, 2)
    assert widget.image_layer.gamma == 0.4
