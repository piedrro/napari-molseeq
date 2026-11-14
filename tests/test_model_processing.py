import numpy as np
import tifffile
from molcore import LocalisationSettings

from molseeq.model_processing import LocalisationJob, localise_channel
from molseeq.session import ImportSettings, ProjectSession, read_recording


def test_detecting_one_alex_frame_returns_logical_index_for_the_widget(
    tmp_path,
):
    movie = np.zeros((5, 13, 26), dtype=np.uint16)
    movie[2, 6, 7] = 200
    path = tmp_path / "alex.tif"
    tifffile.imwrite(path, movie, photometric="minisblack")
    session = ProjectSession()
    session.import_recordings([read_recording(path, ImportSettings("alex"))])
    dataset = session.dataset(path.name)
    channel = session.channel(path.name, "dd")
    rows = localise_channel(
        LocalisationJob(dataset, channel, offset=0, frame=1),
        LocalisationSettings(box_size=5, min_net_gradient=10),
        detect=True,
        fit=False,
    )

    np.testing.assert_array_equal(rows.frame, [1])
    np.testing.assert_array_equal(rows.x, [7])
    np.testing.assert_array_equal(rows.y, [6])
