from concurrent.futures import CancelledError
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from molcore import DriftSettings, StaleResultError
from molcore.staging import state_token

from molseeq.session import ImportSettings, ProjectSession, read_recording
from molseeq.ui import drift


@pytest.fixture
def alex_drift(tmp_path, monkeypatch):
    session = ProjectSession()
    recordings = []
    for index, frame_count in enumerate((5, 6)):
        movie = np.zeros((frame_count, 13, 26), dtype=np.uint16)
        movie[:, 7, [7, 20]] = 1200
        path = tmp_path / f"alex{index}.tif"
        tifffile.imwrite(path, movie, photometric="minisblack")
        recordings.append(read_recording(path, ImportSettings("alex")))
    session.import_recordings(recordings, concatenate=True)
    name = recordings[0].name
    for channel in session.images[name]:
        frames = np.arange(len(session.channel(name, channel)))
        session.localisations[name][channel] = {
            "localisations": np.rec.fromarrays(
                (frames, np.full(len(frames), 7.), np.full(len(frames), 7.)),
                names="frame,x,y",
            ),
            "fitted": True,
        }

    calls = []

    def estimate(snapshot, selections, *, settings, context):
        calls.append((tuple(selections.values()), settings))
        return {
            acquisition_id: SimpleNamespace(
                frames=np.array([1, 3]),
                displacement=np.array([[1, index], [3, index + 2]]),
                backend_version="test-backend",
            )
            for index, acquisition_id in enumerate(selections)
        }

    monkeypatch.setattr(drift.undrifting, "undrift_many", estimate)
    yield session, name, calls
    session.experiment.close()


@pytest.mark.parametrize("method", ["aim", "rcc"])
def test_applies_each_alex_acquisition_drift_at_camera_frames_once(alex_drift, method):
    session, name, calls = alex_drift
    settings = DriftSettings(method=method, segment_frames=2)
    stop = Event()
    revision = session.revision
    expected = {name: state_token(session.dataset(name))}
    result = drift.compute_drift(
        session.image_snapshot(), [name], "dd", settings, stop
    )
    assert calls == [(("dd", "dd"), settings)]
    np.testing.assert_array_equal(session.working_source(name, "ad")[:, 7, 7], 1200)
    ids = session.dataset(name).localizations.query()["id"].copy()
    session.publish_image_correction(result, revision=revision, expected=expected, cancel_event=stop)

    for channel, xs, ys in (
        ("dd", [6, 5, 4, 6, 5, 4], [7, 6, 5, 6, 5, 4]),
        ("ad", [6, 4, 6, 4, 4], [7, 5, 6, 4, 4]),
    ):
        rows = session.localisations[name][channel]["localisations"]
        np.testing.assert_array_equal(rows.x, xs)
        np.testing.assert_array_equal(rows.y, ys)
        for index, frame in enumerate(session.working_source(name, channel)[:]):
            assert np.unravel_index(frame.argmax(), frame.shape) == (ys[index], xs[index])
        np.testing.assert_array_equal(session.channel(name, channel)[:, 7, 7], 1200)

    np.testing.assert_array_equal(session.dataset(name).localizations.query()["id"], ids)
    history = session.dataset(name).processing.settings["molseeq_working_images"]["ad"]
    operation = history["records"][history["active"]]["operation"]
    assert operation["settings"]["method"] == method
    assert operation["reference_channel"] == "dd"
    assert len(operation["estimates"]) == 2

    with pytest.raises(StaleResultError):
        session.publish_image_correction(result, revision=revision, expected=expected, cancel_event=stop)
    np.testing.assert_array_equal(session.localisations[name]["ad"]["localisations"].x, [6, 4, 6, 4, 4])


def test_cancelled_and_stale_corrections_preserve_live_state(alex_drift):
    session, name, _ = alex_drift
    stop = Event()
    revision = session.revision
    expected = {name: state_token(session.dataset(name))}
    snapshot = session.image_snapshot()
    result = drift.compute_drift(snapshot, [name], "dd", DriftSettings(), stop)
    stop.set()
    with pytest.raises(CancelledError):
        drift.compute_drift(snapshot, [name], "dd", DriftSettings(), stop)
    with pytest.raises(CancelledError):
        session.publish_image_correction(result, revision=revision, expected=expected, cancel_event=stop)
    stop.clear()
    session.dataset(name).update_metadata({"edited": True})
    with pytest.raises(StaleResultError):
        session.publish_image_correction(result, revision=revision, expected=expected, cancel_event=stop)
    assert session.revision == revision
    np.testing.assert_array_equal(session.localisations[name]["dd"]["localisations"].x, 7)
    np.testing.assert_array_equal(session.working_source(name, "dd")[:, 7, 7], 1200)


def test_correction_rolls_back_localisation_moves_if_pixels_are_invalid(alex_drift):
    session, name, _ = alex_drift
    stop = Event()
    revision = session.revision
    expected = {name: state_token(session.dataset(name))}
    result = drift.compute_drift(
        session.image_snapshot(), [name], "dd", DriftSettings(), stop
    )
    result[name] = replace(result[name], images={"ad": np.zeros((1, 2, 3))})
    with pytest.raises(ValueError, match="frame mapping"):
        session.publish_image_correction(
            result, revision=revision, expected=expected, cancel_event=stop
        )
    assert state_token(session.dataset(name)) == expected[name]
    assert session.revision == revision
    np.testing.assert_array_equal(session.working_source(name, "ad")[:, 7, 7], 1200)
