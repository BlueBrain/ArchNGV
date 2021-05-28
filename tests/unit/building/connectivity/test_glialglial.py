import sys
from unittest.mock import Mock

import numpy as np
from numpy import testing as npt

import archngv.building.connectivity.glialglial as tested
from archngv.building.connectivity.glialglial import BRANCH_SHIFT, BRANCH_MASK

def _pack_types(pre_type, post_type):
    return (post_type & BRANCH_MASK) | ((pre_type & BRANCH_MASK) << BRANCH_SHIFT)


DATA = {"pre_ids": np.array([
            [0, 1, 2],
            [1, 2, 3]
        ], np.int32),
        "post_ids": np.array([
            [3, 4, 5],
            [2, 3, 4]
        ], np.int32),
        "distances": np.array([[1.0, 1.1, 1.2], [2.1, 2.2, 2.3]]),
        "pre_section_fraction": np.array([0.0, 1.0]),
        "post_section_fraction": np.array([1.0, 0.5]),
        "spine_length": np.array([3.4, 5.6]),
        "pre_position": np.array([[10.0, 10.1, 10.2], [20.1, 20.2, 20.3]]),
        "post_position": np.array([[11.0, 11.1, 11.2], [21.1, 21.2, 21.3]]),
        "branch_type": np.array([_pack_types(1, 3), _pack_types(3, 2)], dtype=np.int8)}


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

    returned = tested.generate_glialglial(None)

    assert len(returned) == 2

    # they must be ordered by target_node_id

    npt.assert_array_equal(returned["source_node_id"], [1, 0])
    npt.assert_array_equal(returned['efferent_section_id'], [2, 1])
    npt.assert_array_equal(returned['efferent_segment_id'], [3, 2])
    npt.assert_allclose(returned['efferent_segment_offset'], [2.2, 1.1])
    npt.assert_array_equal(returned['efferent_section_type'], [3, 1])
    npt.assert_array_equal(returned['efferent_section_pos'], [1.0, 0.0])

    npt.assert_array_equal(returned["target_node_id"], [2, 3])
    npt.assert_array_equal(returned['afferent_section_id'], [3, 4])
    npt.assert_array_equal(returned['afferent_segment_id'], [4, 5])
    npt.assert_allclose(returned['afferent_segment_offset'], [2.3, 1.2])
    npt.assert_array_equal(returned['afferent_section_type'], [2, 3])
    npt.assert_array_equal(returned['afferent_section_pos'], [0.5, 1.0])

    npt.assert_allclose(returned['spine_length'], [5.6, 3.4])

    npt.assert_allclose(
        returned[["efferent_center_x", "efferent_center_y", "efferent_center_z"]],
                        np.array([[20.1, 20.2, 20.3], [10.0, 10.1, 10.2]]))

    npt.assert_allclose(
        returned[["afferent_surface_x", "afferent_surface_y", "afferent_surface_z"]],
                        np.array([[21.1, 21.2, 21.3], [11.0, 11.1, 11.2]]))

    del sys.modules['pytouchreader']
