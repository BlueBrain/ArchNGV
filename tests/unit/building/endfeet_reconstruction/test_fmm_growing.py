import os

import numpy as np
import openmesh
from numpy import testing as npt

from archngv.building.endfeet_reconstruction import fast_marching_method as _fmm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_closest_mesh_nodes():
    mesh_points = np.array([[0.0, 1.0, 3.0], [-1.0, 2.0, 1.0], [5.0, 10.0, 1.0], [1.0, 2.0, 3.0]])

    endfeet_points = np.array(
        [[6.0, 10.0, 2.0], [0.1, 1.2, 3.5], [0.9, 1.5, 3.5], [-2.0, 2.0, 1.0]]
    )

    empty = np.empty(5, dtype=np.int32)
    mesh_vertices = _fmm._find_closest_mesh_nodes(endfeet_points, mesh_points, empty, empty)

    npt.assert_array_equal(mesh_vertices, [2, 0, 3, 1])


def _terminal_plot(mesh_coordinates, mesh_vertices):
    import gnuplotlib as gp

    datasets = [(mesh_coordinates[:, 0], mesh_coordinates[:, 1], {"with": "dots"})]

    coos = mesh_coordinates[mesh_vertices]

    for i in range(len(mesh_vertices)):
        d = (coos[i, 0], coos[i, 1], {"with": f"points pointtype {i + 1}"})
        datasets.append(d)

    gp.plot(*datasets, terminal="dumb 40,20", unset=["grid", "tics"])


def test_closest_mesh_nodes__overlapping():
    """
    All points would overlap at A, but they are distributed so that their
    1-ring neighborhoods do not overlap. The connectivity of the test mesh
    is:
    * -- *
    | \  |
    |  \ |
    |   \|
    * -- *

    Expected Result:
    +---------------------------------+
    |                                 |
    |   .   .  .   F   .   .  .   .   |
    |                                 |
    |   .   A  .   .   .   .  .   .   |
    |   .   .  .   .   .   .  B   .   |
    |                                 |
    |   .   .  .   .   C   .  .   .   |
    |                                 |
    |   .   .  .   .   .   .  .   .   |
    |                                 |
    |   .   .  .   D   .   .  .   E   |
    |   .   .  .   .   .   .  .   .   |
    |                                 |
    |   .   .  .   .   .   .  .   .   |
    |                                 |
    +---------------------------------+
    """
    plane = openmesh.read_trimesh(os.path.join(DATA_DIR, "plane_10x10.obj"))
    neighbors, nn_offsets, xyz = _fmm._mesh_to_flat_arrays(plane)

    endfeet_points = np.array(
        [
            [0.08, 0.09, 0.0],  # 4 - A
            [0.09, 0.10, 0.0],  # 2 - B
            [0.11, 0.11, 0.0],  # 0 - C
            [0.12, 0.12, 0.0],  # 1 - D
            [0.13, 0.13, 0.0],  # 3 - E
            [0.14, 0.14, 0.0],  # 5 - F
        ]
    )

    mesh_vertices = _fmm._find_closest_mesh_nodes(endfeet_points, xyz, neighbors, nn_offsets)
    npt.assert_allclose(mesh_vertices, [72, 67, 55, 34, 38, 84])


def test_closest_mesh_nodes__overlapping_and_normal():
    plane = openmesh.read_trimesh(os.path.join(DATA_DIR, "plane_10x10.obj"))
    neighbors, nn_offsets, xyz = _fmm._mesh_to_flat_arrays(plane)

    endfeet_points = np.array(
        [
            [-1.0, -1.0, 0.0],
            [0.09, 0.09, 0.0],
            [0.10, 0.10, 0.0],
            [0.11, 0.11, 0.0],
            [0.12, 0.12, 0.0],
            [0.13, 0.13, 0.0],
            [0.14, 0.14, 0.0],
            [1.0, 1.0, 1.0],
        ]
    )

    mesh_vertices = _fmm._find_closest_mesh_nodes(endfeet_points, xyz, neighbors, nn_offsets)
    npt.assert_allclose(mesh_vertices, [0, 72, 67, 55, 34, 38, 84, 99])


def test_mesh_to_flat_arrays():
    mesh = openmesh.read_trimesh(os.path.join(DATA_DIR, "cube-minimal.obj"))

    neighbors, offsets, xyz = _fmm._mesh_to_flat_arrays(mesh)

    expected_neighbors = np.array(
        [
            5,
            4,
            6,
            2,
            3,
            1,  # 0:6
            7,
            5,
            0,
            3,  # 6:10
            3,
            0,
            6,
            7,  # 10:14
            1,
            0,
            2,
            7,  # 14:18
            5,
            7,
            6,
            0,  # 18:22
            7,
            4,
            0,
            1,  # 22:26
            7,
            2,
            0,
            4,  # 26:30
            3,
            2,
            6,
            4,
            5,
            1,
        ]
    )  # 30:36
    expected_offsets = np.array([0, 6, 10, 14, 18, 22, 26, 30, 36], dtype=np.int64)
    expected_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    npt.assert_allclose(expected_neighbors, neighbors)
    npt.assert_allclose(expected_offsets, offsets)
    npt.assert_allclose(expected_xyz, xyz)
