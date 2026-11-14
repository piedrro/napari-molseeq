"""Integration checks for snapshot computation and explicit publication."""

from threading import Event

import numpy as np
import pytest
import tifffile
from molcore import StaleResultError, TrackingSettings

from molseeq.session import ImportSettings, ProjectSession, read_recording
from molseeq.trace_views import compute_fixed_traces, publish_traces
from molseeq.ui.tracking import _track_model


class Progress:
    def emit(self, value):
        pass


@pytest.mark.parametrize("concatenate", [False, True])
def test_tracking_computes_detached_results_from_snapshot(
    tmp_path, concatenate
):
    recordings = []
    for index in range(2 if concatenate else 1):
        path = tmp_path / f"movie{index}.tif"
        tifffile.imwrite(
            path, np.ones((3, 9, 9), dtype=np.uint16), photometric="minisblack"
        )
        recordings.append(
            read_recording(path, ImportSettings("single channel"))
        )
    session = ProjectSession()
    session.import_recordings(recordings, concatenate=concatenate)
    name = recordings[0].name
    count = len(session.channel(name, "data"))
    session.localisations[name]["data"]["localisations"] = np.rec.fromarrays(
        (np.arange(count), np.full(count, 4.0), np.full(count, 4.0)),
        names="frame,x,y",
    )
    model = session.dataset(name)
    identifier, is_sequence = session.binding(name, "data")
    result, rows, unlinked = _track_model(
        model.snapshot(),
        identifier,
        is_sequence,
        "data",
        TrackingSettings(search_range=2, min_length=2, frame_clock="channel"),
        True,
        Event(),
        Progress(),
    )
    assert not len(model.tracks)
    assert not len(unlinked)
    np.testing.assert_array_equal(rows[:, 1], np.arange(count))
    receipt = model.apply(result)
    assert len(receipt.track_ids) == 1
    assert len(model.tracks) == 1


def test_trace_computation_does_not_publish_and_rejects_stale_results(
    tmp_path,
):
    path = tmp_path / "movie.tif"
    tifffile.imwrite(
        path, np.ones((2, 9, 9), dtype=np.uint16), photometric="minisblack"
    )
    session = ProjectSession()
    session.import_recordings(
        [read_recording(path, ImportSettings("single channel"))]
    )
    model = session.dataset(path.name)
    results = compute_fixed_traces(
        session,
        np.rec.fromarrays(([4.0], [4.0]), names="x,y"),
        (3, "square", 0, 1, False, False),
    )
    assert not model.photon_sequences.query()
    model.update_metadata({"changed": True})
    with pytest.raises(StaleResultError):
        publish_traces(session, results)
    assert not model.photon_sequences.query()


def test_image_snapshot_keeps_its_original_round_bindings(tmp_path):
    path = tmp_path / "movie.tif"
    tifffile.imwrite(
        path, np.ones((2, 9, 9), dtype=np.uint16), photometric="minisblack"
    )
    session = ProjectSession()
    session.import_recordings(
        [read_recording(path, ImportSettings("single channel"))]
    )
    snapshot = session.image_snapshot()
    session.rename(path.name, "renamed")
    assert tuple(snapshot.images) == (path.name,)
    np.testing.assert_array_equal(
        snapshot.images[path.name]["data"]["data"], 1
    )


def test_localisation_fits_working_pixels_without_changing_raw_model(tmp_path):
    from molcore import ArraySource, LocalisationSettings

    from molseeq.model_processing import LocalisationJob, localise_channel

    path = tmp_path / "movie.tif"
    tifffile.imwrite(
        path, np.zeros((1, 13, 13), dtype=np.uint16), photometric="minisblack"
    )
    session = ProjectSession()
    session.import_recordings(
        [read_recording(path, ImportSettings("single channel"))]
    )
    model = session.dataset(path.name)
    yy, xx = np.indices((13, 13))
    image = (
        100 + 2000 * np.exp(-((xx - 6.0) ** 2 + (yy - 6.0) ** 2) / 2)
    ).astype(np.uint16)
    seeds = np.rec.fromarrays(([0], [6.0], [6.0]), names="frame,x,y")
    rows = localise_channel(
        LocalisationJob(
            model.snapshot(),
            session.channel(path.name, "data"),
            0,
            source=ArraySource(image[None]),
            seeds=seeds,
        ),
        LocalisationSettings(box_size=7),
        detect=False,
        fit=True,
    )
    assert len(rows) == 1
    np.testing.assert_allclose([rows.x[0], rows.y[0]], [6.0, 6.0], atol=0.1)
    assert not model.localizations.query().n_rows
    assert not np.any(session.channel(path.name, "data")[:])


def test_optional_gpufit_uses_prepared_working_pixels(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from molcore import ArraySource, LocalisationSettings

    from molseeq.model_processing import LocalisationJob, localise_channel

    path = tmp_path / "gpu.tif"
    tifffile.imwrite(path, np.zeros((1, 9, 9), dtype=np.uint16), photometric="minisblack")
    session = ProjectSession()
    session.import_recordings([read_recording(path, ImportSettings("single channel"))])
    movie = np.full((1, 9, 9), 100, dtype=np.uint16)
    movie[0, 4, 4] = 150

    def fit(pixels, weights, model, initial, **options):
        assert pixels.shape == (1, 9)
        assert pixels[0, 4] == 50
        assert model == 2
        return initial, np.zeros(1), None, None, None

    monkeypatch.setitem(sys.modules, "pygpufit", SimpleNamespace(
        gpufit=SimpleNamespace(fit=fit, ModelID=SimpleNamespace(GAUSS_2D_ELLIPTIC=2))
    ))
    rows = localise_channel(
        LocalisationJob(
            session.dataset(path.name).snapshot(), session.channel(path.name, "data"), 0,
            source=ArraySource(movie),
            seeds=np.rec.fromarrays(([0], [4.0], [4.0]), names="frame,x,y"),
        ),
        LocalisationSettings(box_size=3),
        detect=False, fit=True, gpu_fit=True,
    )
    np.testing.assert_array_equal(rows.x, [4])
    assert rows.photons[0] > 0
