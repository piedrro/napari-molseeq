import numpy as np
import pytest
import tifffile

from molseeq.session import ImportSettings, ProjectSession, read_recording
from molseeq.trace_views import (
    align_trace_samples,
    expand_export_trace,
    publish_traces,
    stack_export_traces,
    trace_view,
)


def test_efficiency_pairs_only_equal_camera_frames_in_the_same_recording():
    source = {"_camera_frames": np.array([0, 2]), "_acquisition_ids": ("a", "a")}
    reference = {"_camera_frames": np.array([0, 1, 2]), "_acquisition_ids": ("a", "a", "b")}
    np.testing.assert_array_equal(
        align_trace_samples([10, 20], source, reference), [10, np.nan, np.nan]
    )


def test_trace_view_preserves_original_frames_and_existing_metric_names(
    tmp_path,
):
    path = tmp_path / "movie.tif"
    tifffile.imwrite(
        path, np.zeros((5, 4, 6), dtype=np.uint16), photometric="minisblack"
    )
    session = ProjectSession()
    session.import_recordings([read_recording(path, ImportSettings("alex"))])
    values = np.array([4, 8, 15])
    publish_traces(session, {path.name: {"dd": {0: {"spot_sum": values}}}})

    dataset = session.dataset(path.name)
    (sequence,) = dataset.photon_sequences.query()
    np.testing.assert_array_equal(sequence.frames, [0, 2, 4])
    np.testing.assert_array_equal(sequence.photons, values)
    np.testing.assert_array_equal(
        trace_view(session)[path.name]["dd"][0]["spot_sum"], values
    )
    assert sequence.unit == "image_intensity"
    session.images[path.name]["dd"]["data"] = np.ones(
        (3, 4, 3), dtype=np.uint16
    )
    assert not session.dataset(path.name).photon_sequences.query()


def test_core_apertures_project_molseeq_background_sums(tmp_path):
    pytest.importorskip("molcore.photometry", minversion=None)
    from molcore import photometry

    if not hasattr(photometry, "stationary_site_traces"):
        pytest.skip("Requires the candidate MolCore checkout")
    from molseeq.trace_views import compute_fixed_traces

    path = tmp_path / "movie.tif"
    tifffile.imwrite(path, np.arange(1, 26, dtype=np.uint16).reshape(1, 5, 5))
    session = ProjectSession()
    session.import_recordings(
        [read_recording(path, ImportSettings("single channel"))]
    )
    locs = np.rec.fromarrays(([0.0], [0.0]), names="x,y")
    trace = compute_fixed_traces(
        session, locs, (1, "square", 0, 1, True, False)
    )[path.name]["data"][0]

    np.testing.assert_array_equal(trace["spot_sum"], [1])
    np.testing.assert_array_equal(trace["spot_sum_local_bg"], [4])
    np.testing.assert_array_equal(trace["spot_sum_masked_local_bg"], [5])
    np.testing.assert_array_equal(trace["spot_mean_masked_global_bg"], [13.5])


def test_rectangular_alex_export_keeps_empty_cycles_at_recording_boundaries(
    tmp_path,
):
    movies = [tmp_path / name for name in ("first.tif", "second.tif")]
    for path in movies:
        tifffile.imwrite(
            path,
            np.zeros((3, 4, 6), dtype=np.uint16),
            photometric="minisblack",
        )
    session = ProjectSession()
    settings = ImportSettings("alex")
    session.import_recordings(
        [read_recording(path, settings) for path in movies], concatenate=True
    )
    name = movies[0].name
    publish_traces(
        session,
        {
            name: {
                "dd": {0: {"spot_sum": np.array([1, 2, 3, 4])}},
                "aa": {0: {"spot_sum": np.array([5, 6])}},
            }
        },
    )
    traces = trace_view(session)[name]
    columns = [
        expand_export_trace(trace[0]["spot_sum"], trace[0])
        for trace in traces.values()
    ]
    np.testing.assert_array_equal(
        stack_export_traces(columns, axis=1),
        [[1, 5], [2, np.nan], [3, 6], [4, np.nan]],
    )

    import pandas as pd

    from molseeq.ui.trace_export import TraceExportMixin

    exporter = TraceExportMixin()
    exporter.populate_export_dict = lambda: {
        "index": [0, 0], "dataset": [name, name],
        "channel": list(traces), "data": columns,
    }
    destination = tmp_path / "traces.csv"
    exporter.export_traces_csv(export_path=destination)
    exported = pd.read_csv(destination, header=[0, 1, 2])
    assert exported.columns.get_level_values(2).tolist() == list(traces)
    np.testing.assert_array_equal(exported.to_numpy(), [[1, 5], [2, np.nan], [3, 6], [4, np.nan]])


def test_detached_trace_run_publishes_with_an_empty_odd_channel_in_the_first_movie(tmp_path):
    from molcore.staging import experiment_state_token

    from molseeq.trace_views import compute_fixed_traces

    first, second = (tmp_path / name for name in ("first.tif", "second.tif"))
    for path, n_frames in ((first, 1), (second, 3)):
        tifffile.imwrite(
            path, np.full((n_frames, 6, 12), 7, dtype=np.uint16), photometric="minisblack"
        )
    session = ProjectSession()
    session.import_recordings(
        [read_recording(path, ImportSettings("alex")) for path in (first, second)],
        concatenate=True,
    )
    expected = experiment_state_token(session.experiment)
    sites = np.rec.fromarrays(([2.0], [2.0]), names="x,y")
    results = compute_fixed_traces(
        session.image_snapshot(), sites, (1, "square", 0, 1, False, False), expected=expected
    )
    publish_traces(session, results)
    traces = trace_view(session)[first.name]
    columns = [
        expand_export_trace(traces[channel][0]["spot_sum"], traces[channel][0])
        for channel in ("dd", "aa")
    ]
    np.testing.assert_array_equal(
        stack_export_traces(columns, axis=1), [[7, np.nan], [7, 7], [7, np.nan]]
    )
