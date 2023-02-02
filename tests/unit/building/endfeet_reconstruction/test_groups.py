import numpy as np
import pytest
from numpy import testing as npt

from archngv.building.endfeet_reconstruction import groups as _g


@pytest.fixture(scope="module")
def ids():
    return np.random.randint(low=0, high=10, size=6)


@pytest.fixture(scope="module")
def groups():
    return np.array([-1, 10, 22])


@pytest.fixture(scope="module")
def offsets():
    return np.array([0, 2, 3, 6])


@pytest.fixture(scope="module")
def grouped_elements(ids, offsets, groups):
    return _g.GroupedElements(ids, offsets, groups)


def test_grouped_elements_contructor(grouped_elements, ids, offsets, groups):
    npt.assert_array_equal(grouped_elements.ids, ids)
    npt.assert_array_equal(grouped_elements.groups, groups)
    npt.assert_array_equal(grouped_elements._offsets, offsets)


def test_grouped_elements_ids(grouped_elements, ids, offsets, groups):
    npt.assert_array_equal(grouped_elements.get_group_ids(0), ids[0:2])
    npt.assert_array_equal(grouped_elements.get_group_ids(1), ids[2:3])
    npt.assert_array_equal(grouped_elements.get_group_ids(2), ids[3:6])


def test_grouped_elements_iter_assigned_grouped(grouped_elements):
    iter_assigned_groups = grouped_elements.iter_assigned_groups()
    npt.assert_array_equal([g for g, _ in iter_assigned_groups], [10, 22])


def test_group_elements():
    v_group_index = np.array([-1, 3, 3, 3, 2, 2, 1, 0, -1, -1], dtype=np.int32)
    grouped_elements = _g.group_elements(v_group_index)

    npt.assert_array_equal(grouped_elements.groups, [-1, 0, 1, 2, 3])

    npt.assert_array_equal(grouped_elements.get_group_ids(0), [0, 8, 9])  # -1
    npt.assert_array_equal(grouped_elements.get_group_ids(1), [7])  #  0
    npt.assert_array_equal(grouped_elements.get_group_ids(2), [6])  #  1
    npt.assert_array_equal(grouped_elements.get_group_ids(3), [4, 5])  #  2
    npt.assert_array_equal(grouped_elements.get_group_ids(4), [1, 2, 3])  #  3


def test_vertex_to_triangle_groups():
    vertex_groups = np.array([-1, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1])

    triangles = np.array(
        [[1, 5, 4], [3, 1, 4], [1, 3, 2], [1, 6, 5], [2, 7, 9], [6, 7, 8], [8, 9, 7], [10, 11, 1]]
    )

    triangle_groups = _g.vertex_to_triangle_groups(vertex_groups, triangles)

    """
    Vertex groups per triangle. If a row has at least two of the same type,
    the triangle becomes of the same type.
    [ 0,  0,  0] -> 0
    [ 0,  0,  0] -> 0
    [ 0,  0,  0] -> 0
    [ 0,  1,  0] -> 0
    [ 0,  1,  1] -> 1
    [ 1,  1,  1] -> 1
    [ 1,  1,  1] -> 1
    [-1, -1,  0] -> -1
    """

    npt.assert_array_equal(triangle_groups, [0, 0, 0, 0, 1, 1, 1, -1])
