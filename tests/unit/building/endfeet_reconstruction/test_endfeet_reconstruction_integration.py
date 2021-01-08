import pathlib
import tempfile
import pytest
import openmesh

import numpy as np
from numpy import testing as npt

from archngv.core.datasets import EndfootSurfaceMeshes
from archngv.building.endfeet_reconstruction.area_generation import endfeet_area_generation
from archngv.building.exporters.export_endfeet_areas import export_endfeet_areas


_PATH = pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def plane_mesh():
    filepath = str(_PATH / 'data/plane_10x10.obj')
    return openmesh.read_trimesh(filepath)


@pytest.fixture
def endfeet_points():
    return np.array([
        [-1.,  1., 0.], [-1., 0., 0.], [-1., -1., 0.],
        [ 1., -1., 0.], [ 1., 0., 0.], [ 1., 1., 0.]
        ])


@pytest.fixture
def parameters():
    """
    parameters: dict
        The parameters for the algorithms with the following keys:
            - area_distribution [mean, sdev, min, max]
            - thickness_distribution [mean, sdev, min, max]
    """
    return {
        'fmm_cutoff_radius': 1.0,
        'area_distribution': [0.5, 0.1, 0.01, 1.0],
        'thickness_distribution': [0.1, 0.01, 0.01, 1.]
    }


def test_component(plane_mesh, endfeet_points, parameters):

    np.random.seed(0)

    data_generator = endfeet_area_generation(plane_mesh, parameters, endfeet_points)

    with tempfile.NamedTemporaryFile(suffix='.h5') as fd:

        filepath = fd.name
        export_endfeet_areas(filepath, data_generator, len(endfeet_points))

        meshes = EndfootSurfaceMeshes(filepath)

        # without reduction
        expected_areas_initial = [0.6419756, 0.9135803, 0.39506137,
                                  0.5432098, 0.91358054, 0.49382684]

        expected_areas = [0.51851882, 0.59259297, 0.39506137,
                          0.49382717, 0.91358051, 0.44444422]

        npt.assert_allclose(meshes.get('unreduced_surface_area'), expected_areas_initial)
        npt.assert_allclose(meshes.get('surface_area'), expected_areas)

        for i, mesh in enumerate(meshes):
            assert i == mesh.index

