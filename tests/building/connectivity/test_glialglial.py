import sys
from unittest.mock import Mock
import pytest

import numpy as np
import pandas as pd
from numpy import testing as npt

import archngv.building.connectivity.glialglial as _gg


PRE_IDS = [0, 0, 0, 0, 1, 1, 1, 2, 1, 3, 5, 3, 4, 5]
PST_IDS = [1, 1, 2, 3, 4, 4, 4, 4, 3, 1, 0, 2, 5, 1]


@pytest.fixture
def edges():
    return np.column_stack((PRE_IDS, PST_IDS))


class MockCachedDataset:
    def __init__(self, data):
        self.data = data

    def to_nparray(self):
        return self.data


class MockTouchInfo:

    def __init__(self, _):
        pass

    @property
    def touches(self):
        pre_ids = np.array([PRE_IDS]).T
        pst_ids = np.array([PST_IDS]).T
        return {'pre_ids': MockCachedDataset(pre_ids),
                'post_ids': MockCachedDataset(pst_ids)}


def test_edges_from_touchreader(edges):
    """ We mock the TouchInfo class to produce the data we want to test
    """
    sys.modules['pytouchreader'] = Mock(TouchInfo=MockTouchInfo) 

    result_edges = _gg._edges_from_touchreader(None)

    npt.assert_array_equal(result_edges, edges)

    del sys.modules['pytouchreader']


def test_symmetric_connections_and_ids(edges):

    result_edges, result_ids = _gg._symmetric_connections_and_ids(edges)

    expected_edges = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 5],
        [1, 3],
        [1, 4],
        [1, 5],
        [2, 3],
        [2, 4],
        [4, 5],

        # symmetric

        [1, 0],
        [2, 0],
        [3, 0],
        [5, 0],
        [3, 1],
        [4, 1],
        [5, 1],
        [3, 2],
        [4, 2],
        [5, 4],
    ]
    expected_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    npt.assert_array_equal(result_edges, expected_edges)
    npt.assert_array_equal(result_ids, expected_ids)


def test_glialglial_dataframe(edges):

    edges, ids = _gg._symmetric_connections_and_ids(edges)

    result_df = _gg._glialglial_dataframe(edges, ids)

    expected_df = pd.DataFrame({
        'astrocyte_source_id': np.array([1, 2, 3, 5, 0, 3, 4, 5, 0, 3, 4, 0, 1, 2, 1, 2, 5, 0, 1, 4]),
        'astrocyte_target_id': np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]),
        'connection_id': np.array([0, 1, 2, 3, 0, 4, 5, 6, 1, 7, 8, 2, 4, 7, 5, 8, 9, 3, 6, 9])})

    npt.assert_array_equal(result_df.values, expected_df.values)
