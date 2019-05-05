import os
import pytest
import numpy as np

from ..data_endfeetome import Endfeetome
from ...exporters.export_endfeet_areas import export_endfeet_areas


N_ENDFEET = 11


class MockEndfeetome(Endfeetome):

    def __init__(self):

        # substitute the h5 file with a nested dict
        self._fd = {'objects': {}}
        objects = self._fd['objects']

        test_triangles = np.array([[0, 1, 2],
                                   [2, 3, 4],
                                   [4, 5, 0]], dtype=np.uintp)

        for i in range(N_ENDFEET):

            objects['endfoot_' + str(i)] = {
                'points': np.random.random((6, 3)),
                'triangles': test_triangles
            }
    
    def data_generator(self):
        for i in range(N_ENDFEET):
            yield i, self.mesh_points(i), self.mesh_triangles(i)

@pytest.fixture(scope='session')
def endfeetome_path(tmpdir_factory):

    directory_path = tmpdir_factory.getbasetemp()
    path = os.path.join(directory_path, 'endfeetome.h5')
    return path


@pytest.fixture(scope='module')
def endfeetome_mock(endfeetome_path):

    data = MockEndfeetome()
    export_endfeet_areas(endfeetome_path, data.data_generator())
    return data


@pytest.fixture(scope='module')
def endfeetome_data(endfeetome_path, endfeetome_mock): # ensure  mock_data fixture is created first
    return Endfeetome(endfeetome_path)


def test__len__(endfeetome_data):
    assert len(endfeetome_data) == N_ENDFEET


def test__getitem__(endfeetome_data, endfeetome_mock):

    assert not isinstance(endfeetome_data[0], list)

    for endfoot_data, endfoot_mock in zip(endfeetome_data[1: N_ENDFEET], endfeetome_mock[1: N_ENDFEET]):
        assert endfoot_data == endfoot_mock
        assert np.allclose(endfoot_data.points, endfoot_mock.points)
        assert np.allclose(endfoot_data.triangles, endfoot_mock.triangles)

    indices = np.array([1, 5, 2, 0, 8])

    for endfoot_data, endfoot_mock in zip(endfeetome_data[indices], endfeetome_mock[indices]):
        assert endfoot_data == endfoot_mock
        assert np.allclose(endfoot_data.points, endfoot_mock.points)
        assert np.allclose(endfoot_data.triangles, endfoot_mock.triangles)



def test_endfeet_mesh_points(endfeetome_data, endfeetome_mock):
    for i in range(N_ENDFEET):
        assert np.allclose(endfeetome_data.mesh_points(i),
                           endfeetome_data.mesh_points(i))


def test_endfeet_mesh_triangles(endfeetome_data, endfeetome_mock):
    for i in range(N_ENDFEET):
        assert np.allclose(endfeetome_data.mesh_triangles(i),
                           endfeetome_data.mesh_triangles(i))
