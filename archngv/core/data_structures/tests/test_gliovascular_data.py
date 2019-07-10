import os
import h5py
import pytest
import numpy as np

from archngv.core.data_structures.data_gliovascular import GliovascularData

N_ENDFEET = 5


class MockGliovascularData(object):

    endfoot_graph_coordinates = np.random.random((N_ENDFEET, 3))
    endfoot_surface_coordinates = np.random.random((N_ENDFEET, 3))

    def export(self, filepath):
        with h5py.File(filepath, 'w') as fd:
            fd.create_dataset('endfoot_surface_coordinates', data=self.endfoot_surface_coordinates)
            fd.create_dataset('endfoot_graph_coordinates', data=self.endfoot_graph_coordinates)


@pytest.fixture(scope='session')
def gv_data_path(tmpdir_factory):

    directory_path = tmpdir_factory.getbasetemp()

    path = os.path.join(directory_path, 'gliovascular_data.h5')
    return path


@pytest.fixture(scope='module')
def gv_mock(gv_data_path):

    mock_data = MockGliovascularData()
    mock_data.export(gv_data_path)

    return mock_data


@pytest.fixture(scope='module')
def gv_data(gv_data_path, gv_mock):
    return GliovascularData(gv_data_path)


def test_graph_coordinates(gv_data, gv_mock):
    assert np.allclose(gv_data.endfoot_graph_coordinates,
                       gv_mock.endfoot_graph_coordinates)


def test_surface_coordinates(gv_data, gv_mock):
    assert np.allclose(gv_data.endfoot_surface_coordinates,
                       gv_mock.endfoot_surface_coordinates)
