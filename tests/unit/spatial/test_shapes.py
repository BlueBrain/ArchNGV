import numpy
import pytest
from scipy.spatial import ConvexHull

from archngv.spatial import shapes


@pytest.fixture
def convex_polygon():
    points = numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    triangles = numpy.array([[3, 1, 0], [2, 1, 0], [2, 3, 0], [2, 3, 1]])

    face_normals = numpy.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, -1.0, -0.0],
            [-1.0, 0.0, 0.0],
            [0.57735027, 0.57735027, 0.57735027],
        ]
    )

    return shapes.ConvexPolygon(points, triangles)


def test_centroid(convex_polygon):
    expected_centroid = convex_polygon.points.mean(axis=0)

    actual_centroid = convex_polygon.centroid
    assert numpy.allclose(expected_centroid, actual_centroid), "{} : {}".format(
        str(expected_centroid), str(actual_centroid)
    )


def test_volume(convex_polygon):
    assert numpy.isclose(convex_polygon.volume, ConvexHull(convex_polygon.points).volume)


def test_inscribed_sphere(convex_polygon):
    center, radius = convex_polygon.inscribed_sphere

    expected_center = convex_polygon.centroid

    assert numpy.allclose(center, expected_center)

    expected_radius = 0.14433756729740652

    assert numpy.isclose(radius, expected_radius), (radius, expected_radius)


def test_adjacency(convex_polygon):
    adjacency = convex_polygon.adjacency

    expected_adjacency = ({1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2})

    assert adjacency == expected_adjacency, "\n{}\n{}".format(adjacency, expected_adjacency)
