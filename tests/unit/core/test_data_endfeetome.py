import os

import numpy as np
import pytest
from numpy import testing as npt

from archngv.building.exporters import export_endfeet_meshes
from archngv.core.datasets import EndfootMesh, EndfootSurfaceMeshes

# In total there are 6 endfeet but we have data for 4
N_ENDFEET = 6


@pytest.fixture(scope="module")
def indices_per_entry():
    return [0, 1, 2, 3]


@pytest.fixture(scope="module")
def points_per_entry():
    return [
        np.random.random((3, 3)),
        np.random.random((4, 3)),
        np.random.random((5, 3)),
        np.random.random((6, 3)),
    ]


@pytest.fixture(scope="module")
def triangles_per_entry():
    return [
        np.array([[0, 1, 2]], dtype=np.uintp),
        np.array([[0, 1, 2], [2, 3, 0]], dtype=np.uintp),
        np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.uintp),
        np.array([[0, 1, 2], [2, 3, 4], [4, 5, 0]], dtype=np.uintp),
    ]


@pytest.fixture(scope="module")
def areas_per_entry():
    return np.random.random(4)


@pytest.fixture(scope="module")
def initial_areas_per_entry(areas_per_entry):
    return areas_per_entry * 10.0


@pytest.fixture(scope="module")
def thicknesses_per_entry():
    return np.random.random(4)


@pytest.fixture(scope="module")
def endfeet_data(
    indices_per_entry,
    points_per_entry,
    triangles_per_entry,
    areas_per_entry,
    initial_areas_per_entry,
    thicknesses_per_entry,
):
    return [
        EndfootMesh(
            index=index,
            points=points,
            triangles=triangles,
            area=area,
            unreduced_area=unreduced_area,
            thickness=thickness,
        )
        for index, points, triangles, area, unreduced_area, thickness in zip(
            indices_per_entry,
            points_per_entry,
            triangles_per_entry,
            areas_per_entry,
            initial_areas_per_entry,
            thicknesses_per_entry,
        )
    ]


@pytest.fixture(scope="module")
def endfeet_surface_meshes(tmpdir_factory, endfeet_data):
    path = os.path.join(tmpdir_factory.getbasetemp(), "enfeet_areas.h5")

    # write it to file
    export_endfeet_meshes(path, endfeet_data, N_ENDFEET)

    # and load it via the api
    return EndfootSurfaceMeshes(path)


def test__len__(endfeet_surface_meshes):
    assert len(endfeet_surface_meshes) == N_ENDFEET


def test__getitem__(
    endfeet_surface_meshes,
    indices_per_entry,
    points_per_entry,
    triangles_per_entry,
    areas_per_entry,
    thicknesses_per_entry,
):
    assert not isinstance(endfeet_surface_meshes[0], list)

    all_indices = set(indices_per_entry)
    sorted_idx = np.argsort(indices_per_entry)

    n = 0
    for endfoot_id in range(N_ENDFEET):
        endfoot = endfeet_surface_meshes[endfoot_id]
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


def test_endfeet_mesh_points_triangles(
    endfeet_surface_meshes, indices_per_entry, points_per_entry, triangles_per_entry
):
    for i, endfoot_id in enumerate(indices_per_entry):
        npt.assert_allclose(endfeet_surface_meshes.mesh_points(endfoot_id), points_per_entry[i])
        npt.assert_allclose(
            endfeet_surface_meshes.mesh_triangles(endfoot_id), triangles_per_entry[i]
        )


def test_bulk_attributes(
    endfeet_surface_meshes,
    indices_per_entry,
    initial_areas_per_entry,
    areas_per_entry,
    thicknesses_per_entry,
):
    ids = indices_per_entry

    surface_areas = [endfeet_surface_meshes.get("surface_area", index) for index in ids]
    unreduced_surface_areas = [
        endfeet_surface_meshes.get("unreduced_surface_area", index) for index in ids
    ]
    thicknesses = [endfeet_surface_meshes.get("surface_thickness", index) for index in ids]

    npt.assert_allclose(surface_areas, areas_per_entry)
    npt.assert_allclose(unreduced_surface_areas, initial_areas_per_entry)
    npt.assert_allclose(thicknesses, thicknesses_per_entry)
