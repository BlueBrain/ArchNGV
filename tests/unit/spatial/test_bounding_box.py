import pytest
import numpy as np
from archngv.spatial import BoundingBox


@pytest.fixture
def bbox1():

    min_point = np.array([1., 2., 3.])
    max_point = np.array([5., 6., 7.])

    return BoundingBox(min_point, max_point)

@pytest.fixture
def bbox2():

    min_point = np.array([0., 1., 2.])
    max_point = np.array([4., 5., 6.])

    return BoundingBox(min_point, max_point)


def test_from_points_constructor():

    points = np.array([[0., 1., 2.], 
                       [3., 9., 5.],
                       [6., 7., 8.]])

    bbox = BoundingBox.from_points(points)
    expected_bbox = BoundingBox(np.array([0., 1., 2.]),
                                np.array([6., 9., 8.]))

    assert bbox == expected_bbox


def test_from_spheres_constructor():


    points = np.array([[0., 1., 2.], 
                       [3., 9., 5.],
                       [6., 7., 8.]])

    radii = np.array([1., 2., 3.])

    bbox = BoundingBox.from_spheres(points, radii)
    expected_bbox = BoundingBox(np.array([-1., 0., 1.]),
                                np.array([9., 11., 11.]))

    assert bbox == expected_bbox


def test__init__(bbox1):
    assert np.allclose(bbox1._bb, [[1., 2., 3.], [5., 6., 7.]])


def test__eq__(bbox1, bbox2):

    assert bbox1 == bbox1
    assert not bbox1 == bbox2


def test__add__(bbox1, bbox2):

    bbox3 = bbox1 + bbox2
    expected_bbox = BoundingBox(np.array([0., 1., 2.]),
                                np.array([5., 6., 7.]))
    assert bbox3 == expected_bbox


def test_ranges(bbox1):
    assert np.allclose(bbox1.ranges, [[1., 2., 3.], [5., 6., 7.]])


def test_offset(bbox1):
    assert np.allclose(bbox1.offset, bbox1.min_point)


def test_min_point(bbox1):
    assert np.allclose(bbox1.min_point, np.array([1., 2., 3.]))


def test_max_point(bbox1):
    assert np.allclose(bbox1.max_point, np.array([5., 6., 7.]))


def test_center(bbox1):
    assert np.allclose(bbox1.center, [3., 4., 5.])


def test_extent(bbox1):
    assert np.allclose(bbox1.extent, [4., 4., 4.])


def test_layout(bbox1):
    points, edges = bbox1.layout

    expected_points = [[1., 2., 3.],
                       [1., 6., 3.],
                       [1., 2., 7.],
                       [1., 6., 7.],
                       [5., 2., 3.],
                       [5., 6., 3.],
                       [5., 2., 7.],
                       [5., 6., 7.]]

    expected_edges = \
        [(0, 1), (0, 2), (0, 4), (1, 3),
         (1, 5), (2, 3), (2, 6), (3, 7),
         (4, 5), (4, 6), (5, 7), (6, 7)]

    assert np.allclose(expected_points, points)
    assert np.allclose(expected_edges, edges)


def test_volume(bbox1):
    expected_volume = 64.
    assert np.isclose(bbox1.volume, expected_volume)


def test_translate(bbox1):
    bbox1.translate(np.array([-1., -2., -3.]))
    assert np.allclose(bbox1.center, [2., 2., 2.])


def test_scale(bbox1):
    bbox1.scale(np.array([1., 2., 3.]))

    assert np.allclose(bbox1.min_point, [1., 0., -1.])
    assert np.allclose(bbox1.max_point, [5., 8., 11.])


def test_points_inside__inside(bbox1):
    points = np.array([[1.1, 2.1, 3.1],
                       [2., 4.3, 5.5],
                       [4.9, 5.9, 6.9]])
    assert np.all(bbox1.points_inside(points))


def test_points_inside__outside(bbox1):

    points = np.array([[0.8, 2.1, 3.1],
                       [2., 4.3, 5.5],
                       [4.9, 5.9, 6.9]])

    expected = [False, True, False]
    assert not np.all(bbox1.points_inside(points) == expected)


def test_points_inside__border(bbox1):

    points = np.array([[1., 2., 3.], 
                       [5., 6., 7.]])
    assert np.all(bbox1.points_inside(points))


def test_spheres_inside__inside(bbox1):

    points = np.array([[1.1, 2.1, 3.1],
                       [2., 4.3, 5.5],
                       [4.9, 5.9, 6.9]])

    radii = np.array([0.01, 0.01, 0.01])

    assert np.all(bbox1.spheres_inside(points, radii))


def test_spheres_inside__border(bbox1):

    points = np.array([[1.1, 2.1, 3.1],
                       [2., 4.3, 5.5],
                       [4.9, 5.9, 6.9]])

    radii = np.array([1., 1., 1.])

    expected = [False, True, False]
    assert np.all(bbox1.spheres_inside(points, radii) == expected)


def test_spheres_inside__outside(bbox1):

    points = np.array([[100.1, 2.1, 3.1],
                       [200., 4.3, 5.5],
                       [400.9, 5.9, 6.9]])

    radii = np.array([4., 2., 3.])

    assert not np.any(bbox1.spheres_inside(points, radii))

