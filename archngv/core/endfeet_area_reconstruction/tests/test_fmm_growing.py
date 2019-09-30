import os
import numpy as np
from numpy.testing import assert_allclose
from archngv.core.endfeet_area_reconstruction.detail import fmm_growing
import openmesh

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_groups():
    v_group_index = np.array([3, 3, 3, 2, 2, 1, 0, -1, -1], dtype=np.int)
    idx, offsets = fmm_growing._groups(v_group_index)
    assert_allclose(idx, np.array([6, 5, 3, 4, 0, 1, 2, ]))
    assert_allclose(offsets, np.array([0, 1, 2, 4, 7]))


def test__assign_vertex_neighbors():
    mesh = openmesh.read_trimesh(os.path.join(DATA_DIR, 'cube-minimal.obj'))
    neighbors, xyz, offsets = fmm_growing.FastMarchingEikonalSolver._assign_vertex_neighbors(mesh)

    expected_neighbors = np.array([5, 4, 6, 2, 3, 1,  # 0:6
                                   7, 5, 0, 3,        # 6:10
                                   3, 0, 6, 7,        # 10:14
                                   1, 0, 2, 7,        # 14:18
                                   5, 7, 6, 0,        # 18:22
                                   7, 4, 0, 1,        # 22:26
                                   7, 2, 0, 4,        # 26:30
                                   3, 2, 6, 4, 5, 1]) # 30:36
    expected_offsets = np.array([0, 6, 10, 14, 18, 22, 26, 30, 36], dtype=np.int64)
    expected_xyz = np.array([[0., 0., 0.],
                             [0., 0., 1.],
                             [0., 1., 0.],
                             [0., 1., 1.],
                             [1., 0., 0.],
                             [1., 0., 1.],
                             [1., 1., 0.],
                             [1., 1., 1.]], dtype=np.float32)
    assert_allclose(expected_neighbors, neighbors)
    assert_allclose(expected_offsets, offsets)
    assert_allclose(expected_xyz, xyz)
