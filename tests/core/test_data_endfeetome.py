import os
import pytest
import numpy as np
from numpy import testing as npt

from archngv.core.data_endfeet_areas import EndfeetAreas
from archngv.building.exporters.export_endfeet_areas import export_endfeet_areas

# In total there are 6 endfeet but we have data for 4
N_ENDFEET = 6


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
def areas_per_entry():
    return np.random.random(4)


@pytest.fixture(scope='module')
def thicknesses_per_entry():
    return np.random.random(4)


@pytest.fixture(scope='module')
def endfeet_data(indices_per_entry, points_per_entry, triangles_per_entry, areas_per_entry, thicknesses_per_entry):
    return list(zip(indices_per_entry, points_per_entry, triangles_per_entry, areas_per_entry, thicknesses_per_entry))


@pytest.fixture(scope='module')
def endfeet_areas(tmpdir_factory, endfeet_data):

    path = os.path.join(tmpdir_factory.getbasetemp(), 'enfeet_areas.h5')

    # write it to file
    export_endfeet_areas(path, endfeet_data, N_ENDFEET)

    # and load it via the api
    return EndfeetAreas(path)


def test__len__(endfeet_areas):
    assert len(endfeet_areas) == N_ENDFEET


def test__getitem__(endfeet_areas, indices_per_entry, points_per_entry, triangles_per_entry, areas_per_entry, thicknesses_per_entry):

    assert not isinstance(endfeet_areas[0], list)

    all_indices = set(indices_per_entry)
    sorted_idx = np.argsort(indices_per_entry)

    n = 0
    for endfoot_id in range(N_ENDFEET):

        endfoot = endfeet_areas[endfoot_id]
        assert endfoot.index == endfoot_id

        if endfoot_id in all_indices:

            i = sorted_idx[n]

            assert np.allclose(endfoot.points, points_per_entry[i])
            assert np.allclose(endfoot.triangles, triangles_per_entry[i])
            assert np.isclose(endfoot.thickness, thicknesses_per_entry[i])
            assert np.isclose(endfoot.area, areas_per_entry[i])

            n += 1

        else:

            assert endfoot.points.size == 0
            assert endfoot.triangles.size == 0
            assert np.isclose(endfoot.thickness, 0.0)
            assert np.isclose(endfoot.area, 0.0)

def test_endfeet_mesh_points_triangles(endfeet_areas, indices_per_entry, points_per_entry, triangles_per_entry):

    for i, endfoot_id in enumerate(indices_per_entry):

        npt.assert_allclose(endfeet_areas.mesh_points(endfoot_id), points_per_entry[i])
        npt.assert_allclose(endfeet_areas.mesh_triangles(endfoot_id), triangles_per_entry[i])


def test_bulk_attributes(endfeet_areas, indices_per_entry, areas_per_entry, thicknesses_per_entry):

    ids = indices_per_entry

    npt.assert_allclose(endfeet_areas.mesh_surface_areas[ids], areas_per_entry)
    npt.assert_allclose(endfeet_areas.mesh_surface_thicknesses[ids], thicknesses_per_entry)
