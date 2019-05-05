import os
import h5py
import pytest
import numpy as np
from unittest.mock import Mock

from ...exporters.export_gliovascular_connectivity import export_gliovascular_connectivity
from ..connectivity_gliovascular import GliovascularConnectivity


N_ASTROCYTES = 10
N_ENDFEET = 20
N_VASCULATURE_SEGMENTS = N_ENDFEET


class MockGliovascularConnectivity(object):

        def __init__(self):


            self.endfeet_per_astrocyte = \
                np.split(np.arange(N_ENDFEET, dtype=np.uintp), N_ASTROCYTES)

            self.endfoot_to_vasculature = \
                np.arange(N_ENDFEET, dtype=np.uintp)

            self.vasculature_segments_per_astrocyte = self.endfeet_per_astrocyte

            self.astrocyte = Mock(to_endfoot = lambda astro_index: self.endfeet_per_astrocyte[astro_index],
                                  to_vasculature_segment = lambda astro_index: self.vasculature_segments_per_astrocyte[astro_index])

            self.endfoot_to_astrocyte = self._endfoot_to_astrocyte()

            self.endfoot = Mock(to_astrocyte = lambda endf_index: self.endfoot_to_astrocyte[endf_index],
                                to_vasculature_segment = lambda endf_index: self.endfoot_to_vasculature[endf_index])

            self.vasculature_segment = Mock(to_astrocyte = lambda v_index: self.endfoot_to_astrocyte[v_index],
                                            to_endfoot = lambda v_index: v_index)

            self.n_astrocytes = N_ASTROCYTES
            self.n_endfeet = N_ENDFEET


        def _endfoot_to_astrocyte(self):

            res = np.zeros(N_ENDFEET, dtype=np.uintp)

            offset = 0
            for astrocyte_index, endfeet in enumerate(self.endfeet_per_astrocyte):

                n_endfeet = len(endfeet)
                res[offset: offset + n_endfeet] = astrocyte_index
                offset += n_endfeet

            return res


@pytest.fixture(scope='session')
def gv_conn_path(tmpdir_factory):

    directory_path = tmpdir_factory.getbasetemp()

    path = os.path.join(directory_path, 'synaptic_data.h5')
    return path


@pytest.fixture(scope='module')
def gv_conn_mock(gv_conn_path):

    mock_data = MockGliovascularConnectivity()

    export_gliovascular_connectivity(gv_conn_path,
                                     N_ASTROCYTES,
                                     mock_data.endfoot_to_astrocyte,
                                     mock_data.endfoot_to_vasculature)
    return mock_data


@pytest.fixture(scope='module')
def gv_conn_data(gv_conn_path, gv_conn_mock):
    return GliovascularConnectivity(gv_conn_path)


def test_n_astrocytes(gv_conn_data, gv_conn_mock):
    assert gv_conn_mock.n_astrocytes == gv_conn_data.n_astrocytes


def test_n_endfeet(gv_conn_data, gv_conn_mock):
    assert gv_conn_mock.n_endfeet == gv_conn_data.n_endfeet


def test_astrocyte_to_endfoot(gv_conn_data, gv_conn_mock):

    for astrocyte_index in range(N_ASTROCYTES):
        assert np.all(gv_conn_data.astrocyte.to_endfoot(astrocyte_index) == \
                      gv_conn_mock.astrocyte.to_endfoot(astrocyte_index))


def test_astrocyte_to_vasculature_segment(gv_conn_data, gv_conn_mock):
    for astrocyte_index in range(N_ASTROCYTES):
        assert np.all(gv_conn_data.astrocyte.to_vasculature_segment(astrocyte_index) == \
                      gv_conn_mock.astrocyte.to_vasculature_segment(astrocyte_index))


def test_endfoot_to_astrocyte(gv_conn_data, gv_conn_mock):
    for endfoot_index in range(N_ENDFEET):
        print(gv_conn_data.endfoot.to_astrocyte(endfoot_index))
        print(gv_conn_mock.endfoot.to_astrocyte(endfoot_index))
        assert np.all(gv_conn_data.endfoot.to_astrocyte(endfoot_index) == \
                      gv_conn_mock.endfoot.to_astrocyte(endfoot_index))


def test_endfoot_to_vasculature_segment(gv_conn_data, gv_conn_mock):
    for endfoot_index in range(N_ENDFEET):
        assert gv_conn_data.endfoot.to_vasculature_segment(endfoot_index) == \
               gv_conn_mock.endfoot.to_vasculature_segment(endfoot_index)


def test_vasculature_segment_to_endfoot(gv_conn_data, gv_conn_mock):
    for seg_index in range(N_VASCULATURE_SEGMENTS):
        assert gv_conn_data.vasculature_segment.to_endfoot(seg_index) == \
               gv_conn_mock.vasculature_segment.to_endfoot(seg_index)


def test_vasculature_segment_to_astrocyte(gv_conn_data, gv_conn_mock):
    for seg_index in range(N_VASCULATURE_SEGMENTS):
        assert gv_conn_data.vasculature_segment.to_astrocyte(seg_index) == \
               gv_conn_mock.vasculature_segment.to_astrocyte(seg_index)
