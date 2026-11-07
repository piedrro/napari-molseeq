import numpy as np
import tifffile

from molcore import Experiment
from molseeq.session import ImportSettings, ProjectSession, read_recording
from molseeq.trace_views import (
    compute_fixed_traces,
    publish_traces,
    trace_view,
)


def session_with_movie(tmp_path):
    path = tmp_path / "round.tif"
    tifffile.imwrite(
        path, np.ones((5, 9, 18), dtype=np.uint16), photometric="minisblack"
    )
    session = ProjectSession()
    session.import_recordings([read_recording(path, ImportSettings("alex"))])
    return session, path


def define_boxes(session):
    rows = np.rec.fromarrays(
        ([0, 0], [3.0, 5.0], [3.0, 5.0]), names="frame,x,y"
    )
    session.localisation_state["bounding_boxes"] = {
        "localisations": rows,
        "box_size": 5,
        "fitted": False,
        "origin": "manual",
    }
    return rows


def test_molseeq_boxes_own_stable_ids_and_link_sequence_provenance(tmp_path):
    session, path = session_with_movie(tmp_path)
    define_boxes(session)
    ids = session.bounding_box_ids()
    options = (3, "square", 1, 1, True, False)
    results = compute_fixed_traces(
        session, session.bounding_boxes["localisations"], options, box_size=5
    )
    publish_traces(session, results)
    model = session.dataset(path.name)
    sequences = model.photon_sequences.query()
    assert len(sequences) == 8
    assert {sequence.bounding_box_id for sequence in sequences} == set(ids)
    assert all(
        sequence.metadata["extraction"]["settings"]["global_background"]
        for sequence in sequences
    )
    assert all(
        sequence.metadata["bounding_box"]["width"] == 5
        for sequence in sequences
    )
    assert all(
        sequence.metadata["bounding_box"]["metadata"]["origin"] == "manual"
        for sequence in sequences
    )
    np.testing.assert_array_equal(
        trace_view(session)[path.name]["dd"][0]["spot_sum"], [9, 9, 9]
    )

    edited = session.bounding_boxes.copy()
    edited["localisations"].x[0] = 4
    session.localisation_state["bounding_boxes"] = edited
    assert session.bounding_box_ids() == ids
    assert not trace_view(session)
    assert (
        len(
            session.dataset(path.name).photon_sequences.query(
                include_stale=True
            )
        )
        == 8
    )
    assert all(
        session.dataset(path.name).photon_sequences.is_stale(sequence.id)
        for sequence in sequences
    )


def test_boxes_are_copied_to_new_rounds_and_roundtrip_with_the_experiment(
    tmp_path,
):
    session, path = session_with_movie(tmp_path)
    rows = define_boxes(session)
    second = tmp_path / "second.tif"
    tifffile.imwrite(
        second, np.ones((3, 9, 18), dtype=np.uint16), photometric="minisblack"
    )
    session.import_recordings([read_recording(second, ImportSettings("alex"))])
    ids = session.bounding_box_ids()
    assert all(
        tuple(model.bounding_boxes) == ids
        for model in session.experiment.datasets.values()
    )
    snapshot = tmp_path / "surface.h5"
    session.experiment.save(snapshot, image_policy="none")
    restored = Experiment.open(snapshot)
    session.experiment = restored
    assert session.bounding_box_ids() == ids
    np.testing.assert_array_equal(
        session.bounding_boxes["localisations"], rows
    )
    assert session.bounding_boxes["box_size"] == 5


def test_sequence_provenance_retains_the_working_image_recipe(tmp_path):
    session, path = session_with_movie(tmp_path)
    define_boxes(session)
    source = session.channel(path.name, "dd")
    session.replace_working_images(
        {(path.name, "dd"): source[:] + 1},
        operation={
            "method": "temporal_filter",
            "radius": 2,
            "reduction": "median",
        },
    )
    output = compute_fixed_traces(
        session,
        session.bounding_boxes["localisations"],
        (3, "square", 1, 1, False, False),
        box_size=5,
    )
    publish_traces(session, output)
    destination = tmp_path / "traces.h5"
    session.experiment.save(destination, image_policy="none")
    restored = Experiment.open(destination)
    dataset = next(iter(restored.datasets.values()))
    sequence = dataset.photon_sequences.query(channel="dd")[0]
    provenance = sequence.metadata["extraction"]
    history_id = provenance["working_source"]["history_id"]
    operation = dataset.processing.settings["molseeq_working_images"]["dd"][
        "records"
    ][history_id]["operation"]
    assert operation == {
        "method": "temporal_filter",
        "radius": 2,
        "reduction": "median",
    }
    assert provenance["software"]["molcore"]
    assert sequence.metadata["bounding_box"]["metadata"]["created_at"]
    assert sequence.run_id in dataset.processing.settings["photon_sequences"]


def test_reordering_boxes_preserves_ids_and_projects_traces_in_the_new_order(
    tmp_path,
):
    session, path = session_with_movie(tmp_path)
    define_boxes(session)
    first, second = session.bounding_box_ids()
    result = compute_fixed_traces(
        session,
        session.bounding_boxes["localisations"],
        (3, "square", 1, 1, False, False),
        box_size=5,
    )
    publish_traces(session, result)
    entry = session.bounding_boxes.copy()
    entry["localisations"] = entry["localisations"][::-1]
    entry["_box_ids"] = [second, first]
    session.localisation_state["bounding_boxes"] = entry
    assert session.bounding_box_ids() == (second, first)
    assert len(session.dataset(path.name).photon_sequences.query()) == 8
    np.testing.assert_array_equal(
        trace_view(session)[path.name]["dd"][0]["spot_x"], [5, 5, 5]
    )
