import numpy as np
import pytest
import tifffile
from astropy.io import fits

from molseeq.session import ImportSettings, ProjectSession, read_recording


def movie(tmp_path, name, frames=5, width=6):
    path = tmp_path / name
    pixels = np.arange(frames * 4 * width, dtype=np.uint16).reshape(
        frames, 4, width
    )
    tifffile.imwrite(path, pixels, photometric="minisblack")
    return path, pixels


def test_alex_retains_odd_last_frame_and_acceptor_first_layout(tmp_path):
    path, pixels = movie(tmp_path, "odd.tif")
    recording = read_recording(
        path, ImportSettings("alex", "acceptor-donor", "acceptor")
    )
    session = ProjectSession()
    session.import_recordings([recording])

    dd = session.channel(path.name, "dd")
    aa = session.channel(path.name, "aa")
    np.testing.assert_array_equal(dd.source_frames, [1, 3])
    np.testing.assert_array_equal(aa.source_frames, [0, 2, 4])
    np.testing.assert_array_equal(dd[:], pixels[1::2, :, 3:])
    np.testing.assert_array_equal(aa[:], pixels[::2, :, :3])


def test_fret_fits_uses_two_rois_on_the_same_limited_camera_frames(tmp_path):
    path = tmp_path / "fret.fits"
    pixels = np.arange(4 * 4 * 6, dtype=np.uint16).reshape(4, 4, 6)
    fits.writeto(path, pixels)
    session = ProjectSession()
    session.import_recordings([read_recording(path, ImportSettings("fret", limit=2))])
    donor = session.channel(path.name, "donor")
    acceptor = session.channel(path.name, "acceptor")
    assert donor.acquisition is acceptor.acquisition
    np.testing.assert_array_equal(donor[:], pixels[:2, :, :3])
    np.testing.assert_array_equal(acceptor[:], pixels[:2, :, 3:])


def test_append_binds_independent_channels_and_rename_preserves_dataset_id(
    tmp_path,
):
    first, _ = movie(tmp_path, "first.tif")
    second, pixels = movie(tmp_path, "second.tif", frames=3)
    session = ProjectSession()
    session.import_recordings([read_recording(first, ImportSettings("dd"))])
    dataset_id = session.dataset(first.name).id
    session.import_recordings(
        [read_recording(second, ImportSettings("aa"))], append_to=first.name
    )
    session.rename(first.name, "round 1")

    assert session.dataset("round 1").id == dataset_id
    assert len(session.dataset("round 1").acquisitions) == 2
    np.testing.assert_array_equal(session.channel("round 1", "aa")[:], pixels)


def test_concatenate_preserves_recording_identity_and_rejects_bad_batches(
    tmp_path,
):
    first, pixels = movie(tmp_path, "first.tif", frames=3)
    second, other = movie(tmp_path, "second.tif", frames=2)
    third, _ = movie(tmp_path, "different-width.tif", width=8)
    settings = ImportSettings("single channel")
    session = ProjectSession()

    with pytest.raises(ValueError, match="geometry"):
        session.import_recordings(
            [
                read_recording(path, settings)
                for path in (first, second, third)
            ],
            concatenate=True,
        )
    assert not session.experiment.datasets

    session.import_recordings(
        [read_recording(path, settings) for path in (first, second)],
        concatenate=True,
    )
    view = session.channel(first.name, "data")
    np.testing.assert_array_equal(view[:], np.concatenate([pixels, other]))
    assert (
        view.frame_info(2).acquisition_id != view.frame_info(3).acquisition_id
    )
    assert view.frame_info(3).camera_frame == 0


def test_working_images_keep_original_camera_pixels(tmp_path):
    path, pixels = movie(tmp_path, "round.tif")
    session = ProjectSession()
    session.import_recordings([read_recording(path, ImportSettings("dd"))])
    channel = session.images[path.name]["dd"]
    channel["data"] = pixels + 10
    channel["sequence_label"] = "A"

    np.testing.assert_array_equal(
        session.images[path.name]["dd"]["data"], pixels + 10
    )
    np.testing.assert_array_equal(session.channel(path.name, "dd")[:], pixels)
    assert session.dataset(path.name).metadata["sequence_label"] == "A"


def test_localisation_views_map_alex_frames_and_keep_filtered_ids(tmp_path):
    path, _ = movie(tmp_path, "round.tif")
    session = ProjectSession()
    session.import_recordings([read_recording(path, ImportSettings("alex"))])
    rows = np.rec.fromarrays(
        ([0, 1], [1.0, 2.0], [2.0, 2.0]), names="frame,x,y"
    )
    channel = session.localisations[path.name]["dd"]
    channel["localisations"] = rows

    model = session.dataset(path.name)
    before = model.localizations.query(channel="dd", space="raw")
    np.testing.assert_array_equal(before["frame"], [0, 2])
    channel["localisations"] = rows[1:]
    after = model.localizations.query(channel="dd", space="raw")
    np.testing.assert_array_equal(after["id"], before["id"][1:])
    np.testing.assert_array_equal(channel["localisations"].frame, [1])


def test_combined_export_places_working_alex_pixels_in_camera_order(tmp_path):
    path, pixels = movie(tmp_path, "round.tif")
    session = ProjectSession()
    session.import_recordings(
        [
            read_recording(
                path, ImportSettings("alex", first_excitation="acceptor")
            )
        ]
    )
    dd = session.images[path.name]["dd"]
    dd["data"] = dd["data"] + 100

    expected = pixels.copy()
    expected[1::2, :, :3] += 100
    np.testing.assert_array_equal(
        list(session.camera_frames(path.name, "alex")), expected
    )


def test_concatenation_keeps_the_processed_prefix(tmp_path):
    first, original = movie(tmp_path, "first.tif")
    second, tail = movie(tmp_path, "second.tif", frames=2)
    settings = ImportSettings("single channel")
    session = ProjectSession()
    session.import_recordings([read_recording(first, settings)])
    session.images[first.name]["data"]["data"] = original + 3
    session.import_recordings(
        [read_recording(second, settings)],
        append_to=first.name,
        concatenate=True,
    )
    np.testing.assert_array_equal(
        session.working_source(first.name, "data")[:],
        np.concatenate([original + 3, tail]),
    )
