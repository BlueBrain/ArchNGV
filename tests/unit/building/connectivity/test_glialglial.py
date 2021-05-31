import sys
from unittest.mock import Mock

import numpy as np
from numpy import testing as npt

import archngv.building.connectivity.glialglial as tested
from archngv.building.connectivity.glialglial import BRANCH_SHIFT, BRANCH_MASK


EXPECTED_COLUMNS = {
    'source_node_id',
    'efferent_section_id',
    'efferent_segment_id',
    'target_node_id',
    'afferent_section_id',
    'afferent_segment_id',
    'efferent_segment_offset',
    'afferent_segment_offset',
    'efferent_section_pos',
    'afferent_section_pos',
    'spine_length',
    'efferent_center_x',
    'efferent_center_y',
    'efferent_center_z',
    'afferent_surface_x',
    'afferent_surface_y',
    'afferent_surface_z',
    'efferent_section_type',
    'afferent_section_type'
}


def _pack_types(pre_type, post_type):
    # TD reduces the section type by one
    pre_type -= 1
    post_type -= 1
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


EMPTY_DATA = {
        "pre_ids": np.empty((0, 3), np.int32),
        "post_ids": np.empty((0, 3), np.int32),
        "distances": np.empty((0, 3), dtype=np.float32),
        "pre_section_fraction": np.empty(0, dtype=np.float32),
        "post_section_fraction": np.empty(0, dtype=np.float32),
        "spine_length": np.empty(0, dtype=np.float32),
        "pre_position": np.empty((0, 3), dtype=np.float32),
        "post_position": np.empty((0, 3), dtype=np.float32),
        "branch_type": np.empty(0, dtype=np.int8)
}


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


class EmptyMockTouchInfo:

    def __init__(self, _):
        pass

    @property
    def touches(self):
        return MockTouches(EMPTY_DATA)


def test_glialglial_dataframe():
    sys.modules['pytouchreader'] = Mock(TouchInfo=MockTouchInfo)

    df = tested.generate_glialglial(None)

    assert len(df) == 2
    assert set(df.columns) == EXPECTED_COLUMNS

    # they must be ordered by target_node_id

    npt.assert_array_equal(df["source_node_id"], [1, 0])
    npt.assert_array_equal(df['efferent_section_id'], [2, 1])
    npt.assert_array_equal(df['efferent_segment_id'], [3, 2])
    npt.assert_allclose(df['efferent_segment_offset'], [2.2, 1.1])
    npt.assert_array_equal(df['efferent_section_type'], [3, 1])
    npt.assert_array_equal(df['efferent_section_pos'], [1.0, 0.0])

    npt.assert_array_equal(df["target_node_id"], [2, 3])
    npt.assert_array_equal(df['afferent_section_id'], [3, 4])
    npt.assert_array_equal(df['afferent_segment_id'], [4, 5])
    npt.assert_allclose(df['afferent_segment_offset'], [2.3, 1.2])
    npt.assert_array_equal(df['afferent_section_type'], [2, 3])
    npt.assert_array_equal(df['afferent_section_pos'], [0.5, 1.0])

    npt.assert_allclose(df['spine_length'], [5.6, 3.4])

    npt.assert_allclose(
        df[["efferent_center_x", "efferent_center_y", "efferent_center_z"]],
                        np.array([[20.1, 20.2, 20.3], [10.0, 10.1, 10.2]]))

    npt.assert_allclose(
        df[["afferent_surface_x", "afferent_surface_y", "afferent_surface_z"]],
                        np.array([[21.1, 21.2, 21.3], [11.0, 11.1, 11.2]]))

    del sys.modules['pytouchreader']


def test_glialglial_dataframe__empty():
    sys.modules['pytouchreader'] = Mock(TouchInfo=EmptyMockTouchInfo)

    df = tested.generate_glialglial(None)
    arr = df.to_numpy()
    assert arr.shape == (0, len(EXPECTED_COLUMNS))
    assert set(df.columns) == EXPECTED_COLUMNS

    del sys.modules['pytouchreader']
