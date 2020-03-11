import os
import h5py
import pytest
import tempfile
import numpy as np
from numpy import testing as npt
from unittest.mock import Mock

from archngv.core.connectivity_gliovascular import GliovascularConnectivity
from archngv.building.exporters.export_gliovascular_connectivity import export_gliovascular_connectivity


N_ENDFEET = 20
N_ASTROCYTES = 10


@pytest.fixture(scope='module')
def astrocyte_to_endfoot():
    """ ids of endfeet per astrocyte """
    #          0          1        2      3     4      5    6            7               8         9
    return [[11, 14], [0, 1, 2], [3, 4], [5], [6, 7], [], [8, 9], [10, 12, 13, 15], [16, 17, 18], [19]]


@pytest.fixture(scope='module')
def endfoot_to_astrocyte():
    """ The corresponding astrocyte for each endfoot """
    #       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
    return [1, 1, 1, 2, 2, 3, 4, 4, 6, 6, 7, 0, 7, 7, 0, 7, 8, 8, 8, 9]


@pytest.fixture(scope='module')
def endfoot_to_vasculature():
    """ The vasculature section and segment id for each endfoot """
    return {'section_id': [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 1, 2, 3, 4],
            'segment_id': [1, 2, 4, 1, 2, 3, 3, 5, 1, 2, 1, 0, 0, 0, 1, 2, 3, 4, 1, 1]}


@pytest.fixture(scope='module')
def gv_connectivity(endfoot_to_astrocyte, endfoot_to_vasculature):

    with tempfile.NamedTemporaryFile(suffix='.h5') as fd:

        path = fd.name


        e2v = np.column_stack((endfoot_to_vasculature['section_id'],
                               endfoot_to_vasculature['segment_id']))

        export_gliovascular_connectivity(path,
                                         N_ASTROCYTES,
                                         endfoot_to_astrocyte,
                                         e2v)

        return GliovascularConnectivity(path)


def test_lengths(gv_connectivity):
    npt.assert_equal(gv_connectivity.n_astrocytes, N_ASTROCYTES)
    npt.assert_equal(gv_connectivity.n_endfeet, N_ENDFEET)


def test_astrocyte_to_endfoot(gv_connectivity, astrocyte_to_endfoot):
    for astrocyte_index, expected_endfeet_ids in enumerate(astrocyte_to_endfoot):
        ids = gv_connectivity.astrocyte.to_endfoot(astrocyte_index)
        npt.assert_array_equal(ids, expected_endfeet_ids)


def test_endfoot_to_astrocyte(gv_connectivity, endfoot_to_astrocyte):
    for endfoot_index, expected in enumerate(endfoot_to_astrocyte):
        ids = gv_connectivity.endfoot.to_astrocyte(endfoot_index)
        npt.assert_array_equal(ids, expected)


def test_endfoot_to_vasculature(gv_connectivity, endfoot_to_vasculature):

    section_ids = endfoot_to_vasculature['section_id']
    segment_ids = endfoot_to_vasculature['segment_id']

    for endfoot_index, expected in enumerate(zip(section_ids, segment_ids)):
        ids = gv_connectivity.endfoot.to_vasculature_segment(endfoot_index)
        npt.assert_array_equal(ids, expected)
