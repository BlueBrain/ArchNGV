import numpy as np
from ..cell_grid import Point


def test_point():

    p_array_1 = np.array([0.1, 0.2, 0.3])

    point_1 = Point(p_array_1)

    assert np.isclose(point_1[0], 0.1)
    assert np.isclose(point_1[1], 0.2)
    assert np.isclose(point_1[2], 0.3)

    p_array_2 = np.array([0.10, 0.20, 0.30])
    point_2 = Point(p_array_2)
    assert point_1 == point_2

    assert hash(point_1) == hash(point_2)


def point_array():

    z = np.arange(-10, 10.1, 0.1)
    x = y = np.zeros(len(z), dtype=np.float)

    return np.column_stack((x, y, z))


def test_grid_point_registry_constructior():

    cutoff_distance = 1.2
    points = point_array()

    grid = GridPointRegistry(points, cutoff_distance)
