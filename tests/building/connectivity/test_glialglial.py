import sys
from unittest.mock import Mock

import numpy as np
from numpy import testing as npt

import archngv.building.connectivity.glialglial as tested

DATA = {"pre_ids": np.array([[0, 0, 0], [1, 1, 1]], np.int32),
        "post_ids": np.array([[3, 3, 3], [2, 2, 2]], np.int32),
        "distances": np.array([[1.0, 1.1, 1.2], [2.1, 2.2, 2.3]]),
        "pre_section_fraction": np.array([0.0, 1.0]),
        "post_section_fraction": np.array([0.0, 1.0]),
        "spine_length": np.array([0.0, 1.0]),
        "pre_position": np.array([[10.0, 10.1, 10.2], [20.1, 20.2, 20.3]]),
        "post_position": np.array([[11.0, 11.1, 11.2], [21.1, 21.2, 21.3]]),
        "branch_type": np.array([0, 0], dtype=np.int8)}


class MockCachedDataset:
    def __init__(self, data):
        self.data = data

    def to_nparray(self):
        return self.data


class MockTouches:
    def __init__(self, data):
        self.data = {k: MockCachedDataset(v) for k, v in data.items()}

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return 2


class MockTouchInfo:

    def __init__(self, _):
        pass

    @property
    def touches(self):
        return MockTouches(DATA)


def test_glialglial_dataframe():
    sys.modules['pytouchreader'] = Mock(TouchInfo=MockTouchInfo)

    returned = tested.generate_glialglial("unused")

    assert sorted(list(returned)) == sorted(["pre_id", "pre_section_id", "pre_segment_id",
                                             "post_id", "post_section_id", "post_segment_id",
                                             "distances_x", "distances_y", "distances_z",
                                             "pre_section_fraction", "post_section_fraction",
                                             "spine_length", "efferent_center_x",
                                             "efferent_center_y", "efferent_center_z",
                                             "afferent_surface_x", "afferent_surface_y",
                                             "afferent_surface_z", "branch_type"])

    assert len(returned) == 2

    # correctly ordered
    npt.assert_equal(returned["post_id"].to_numpy(), [2, 3])
    npt.assert_equal(returned["pre_id"].to_numpy(), [1, 0])
    npt.assert_allclose(returned[["efferent_center_x", "efferent_center_y", "efferent_center_z"]],
                        np.array([[20.1, 20.2, 20.3], [10.0, 10.1, 10.2]]))
    npt.assert_allclose(returned[["afferent_surface_x", "afferent_surface_y",
                                  "afferent_surface_z"]],
                        np.array([[21.1, 21.2, 21.3], [11.0, 11.1, 11.2]]))
    del sys.modules['pytouchreader']
