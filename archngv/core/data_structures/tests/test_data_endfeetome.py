import os
import pytest
import numpy as np

from archngv.core.data_structures.data_endfeet_areas import EndfeetAreas
from archngv.core.exporters.export_endfeet_areas import export_endfeet_areas


N_ENDFEET = 11


class MockEndfeetAreas(EndfeetAreas):

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
def endfeet_areas_path(tmpdir_factory):

    directory_path = tmpdir_factory.getbasetemp()
    path = os.path.join(directory_path, 'endfeet_areas.h5')
    return path


@pytest.fixture(scope='module')
def endfeet_areas_mock(endfeet_areas_path):

    data = MockEndfeetAreas()
    export_endfeet_areas(endfeet_areas_path, data.data_generator())
    return data


@pytest.fixture(scope='module')
def endfeet_areas_data(endfeet_areas_path, endfeet_areas_mock): # ensure  mock_data fixture is created first
    return EndfeetAreas(endfeet_areas_path)


def test__len__(endfeet_areas_data):
    assert len(endfeet_areas_data) == N_ENDFEET


def test__getitem__(endfeet_areas_data, endfeet_areas_mock):

    assert not isinstance(endfeet_areas_data[0], list)

    for endfoot_data, endfoot_mock in zip(endfeet_areas_data[1: N_ENDFEET], endfeet_areas_mock[1: N_ENDFEET]):
        assert endfoot_data == endfoot_mock
        assert np.allclose(endfoot_data.points, endfoot_mock.points)
        assert np.allclose(endfoot_data.triangles, endfoot_mock.triangles)

    indices = np.array([1, 5, 2, 0, 8])

    for endfoot_data, endfoot_mock in zip(endfeet_areas_data[indices], endfeet_areas_mock[indices]):
        assert endfoot_data == endfoot_mock
        assert np.allclose(endfoot_data.points, endfoot_mock.points)
        assert np.allclose(endfoot_data.triangles, endfoot_mock.triangles)



def test_endfeet_mesh_points(endfeet_areas_data, endfeet_areas_mock):
    for i in range(N_ENDFEET):
        assert np.allclose(endfeet_areas_data.mesh_points(i),
                           endfeet_areas_data.mesh_points(i))


def test_endfeet_mesh_triangles(endfeet_areas_data, endfeet_areas_mock):
    for i in range(N_ENDFEET):
        assert np.allclose(endfeet_areas_data.mesh_triangles(i),
                           endfeet_areas_data.mesh_triangles(i))
