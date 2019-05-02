import pytest
import numpy as np
from ..synaptic_seeds import PointCloud


def point_array():

    z = np.arange(-10, 10.1, 0.1)
    x = y = np.zeros(len(z), dtype=np.float)

    return np.column_stack((x, y, z))


@pytest.fixture
def point_cloud():

    cutoff_radius = 5.0
    removal_radius = 1.0

    return PointCloud(point_array(), cutoff_radius, removal_radius)


def test_constructor(point_cloud):

    points = point_array()

    assert point_cloud.number_of_points == len(points)
    assert np.isclose(point_cloud.default_radius_of_influence, 5.0)
    assert np.isclose(point_cloud.default_removal_radius, 1.0)


def at_least_n_points_around(point_cloud):

    point = np.array([-11., -11., -11.])

    assert point_cloud.at_least_n_points_around(point, 0.0, 0)
    assert not point_cloud.at_least_n_points_around(point, 0.0, 1)

    assert point_cloud.at_least_n_points_around(point, 1.0, 1)
    assert not point_cloud.at_least_n_points_around(point, 1.0, 2)

    assert point_cloud.at_least_n_points_around(point, 2.0, 1)
    assert point_cloud.at_least_n_points_around(point, 2.0, 2)
    assert not point_cloud.at_least_n_points_around(point, 2.0, 3)


def test_average_direction():

    points = np.array([[2., 0., 0.],
                       [3., 2., 1.]])

    point = np.array([1., 0., 0.])

    influence_radius = 5.0
    removal_radius = 0.1
    point_cloud = PointCloud(points, influence_radius, removal_radius)

    expected_direction = np.array([0.91287093, 0.36514837, 0.18257419])
    direction = point_cloud.average_direction(point)

    assert direction is not None
    assert np.allclose(direction, expected_direction)

    direction = point_cloud.average_direction(point, radius_of_influence=0.1)
    assert direction is None


def test_remove_points_around(point_cloud):

    point = np.array([0.1, 0.1, 0.1])

    removal_radius = 0.2

    all_ids = point_cloud.available_ids

    removed_ids_1 = point_cloud.remove_points_around(point, removal_radius)
    assert removed_ids_1 not in point_cloud

    remaining_ids = np.setdiff1d(all_ids, removed_ids_1)
    assert remaining_ids in point_cloud

    removed_ids_2 = point_cloud.remove_points_around(point, removal_radius=0.3)
    assert removed_ids_2 not in point_cloud

    remaining_ids = np.setdiff1d(remaining_ids, removed_ids_2)
    assert remaining_ids in point_cloud
