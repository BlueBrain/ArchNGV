import os
import pytest
import numpy as np

from archngv.core.data_endfeet_areas import EndfeetAreas
from archngv.building.exporters.export_endfeet_areas import export_endfeet_areas


N_ENDFEET = 4


@pytest.fixture(scope='module')
def indices_per_entry():
    return [5, 2, 0, 1]


@pytest.fixture(scope='module')
def points_per_entry():
    return [
        np.random.random((3, 3)),
        np.random.random((4, 3)),
        np.random.random((5, 3)),
        np.random.random((6, 3))]


@pytest.fixture(scope='module')
def triangles_per_entry():
    return [
        np.array([[0, 1, 2]], dtype=np.uintp),
        np.array([[0, 1, 2],
                  [2, 3, 0]], dtype=np.uintp),
        np.array([[0, 1, 2],
                  [1, 2, 3],
                  [2, 3, 4]], dtype=np.uintp),
        np.array([[0, 1, 2],
                  [2, 3, 4],
                  [4, 5, 0]], dtype=np.uintp)]


@pytest.fixture(scope='module')
def thicknesses_per_entry():
    return np.random.random(N_ENDFEET)


@pytest.fixture(scope='module')
def endfeet_areas(tmpdir_factory, indices_per_entry, points_per_entry, triangles_per_entry, thicknesses_per_entry):
    path = os.path.join(tmpdir_factory.getbasetemp(), 'enfeet_areas.h5')
    export_endfeet_areas(path, zip(
        indices_per_entry,
        points_per_entry,
        triangles_per_entry,
        thicknesses_per_entry))
    return EndfeetAreas(path)


def test__len__(endfeet_areas):
    assert len(endfeet_areas) == N_ENDFEET


def test__getitem__(endfeet_areas, indices_per_entry, points_per_entry, triangles_per_entry, thicknesses_per_entry):

    assert not isinstance(endfeet_areas[0], list)

    for i, endfoot_id in enumerate(indices_per_entry):
        endfoot = endfeet_areas[endfoot_id]
        assert endfoot.index == indices_per_entry[i]
        assert np.allclose(endfoot.points, points_per_entry[i])
        assert np.allclose(endfoot.triangles, triangles_per_entry[i])
        assert np.isclose(endfoot.thickness, thicknesses_per_entry[i])


def test_endfeet_mesh_points(endfeet_areas, indices_per_entry, points_per_entry):
    for i, endfoot_id in enumerate(indices_per_entry):
        assert np.allclose(endfeet_areas.mesh_points(endfoot_id), points_per_entry[i])


def test_endfeet_mesh_triangles(endfeet_areas, indices_per_entry, triangles_per_entry):
    for i, endfoot_id in enumerate(indices_per_entry):
        assert np.allclose(endfeet_areas.mesh_triangles(endfoot_id), triangles_per_entry[i])
