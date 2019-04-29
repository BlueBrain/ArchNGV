import numpy as np

from ..synaptic_seeds import PointCloud


def point_array():

    z = np.arange(-10, 10.1, 0.1)
    x = y = np.zeros(len(z), dtype=np.float)

    return np.column_stack((x, y, z))


def test_constructor():

    points = point_array()

    influence_radius = 1.2
    point_cloud = PointCloud(points, influence_radius)


    assert point_cloud._size == len(points)
    assert point_cloud.radius_of_influence == influence_radius


def test_coordinates():

    points = point_array()

    influence_radius = 1.2
    point_cloud = PointCloud(points, influence_radius)

    print(point_cloud.coordinates, points)
    # assert np.allclose(point_cloud.coordinates, points)
