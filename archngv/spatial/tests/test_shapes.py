import pytest

import numpy
from scipy.spatial import ConvexHull
from .. import shapes


import archngv.math_utils as mt


@pytest.fixture
def convex_polygon():
    points = numpy.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])

    face_vertices = numpy.array([[3, 1, 0], [2, 1, 0], [2, 3, 0], [2, 3, 1]])

    face_normals = numpy.array([[0., 0., -1.],
                                [0., -1., -0.],
                                [-1., 0., 0.],
                                [0.57735027, 0.57735027, 0.57735027]])

    return shapes.ConvexPolygon(points, face_vertices)


def assert_equal_triangles(tris1, tris2):

    tris1 = numpy.sort(tris1, axis=1)
    tris2 = numpy.sort(tris2, axis=1)

    tris1 = tris1[numpy.lexsort(tris1.T)]
    tris2 = tris2[numpy.lexsort(tris2.T)]

    print(tris1)
    print(tris2)

    numpy.testing.assert_allclose(tris1, tris2)


def circle_inscribed_polygon(n_points):

    thetas = numpy.linspace(0., 1.8 * numpy.pi, n_points)
    cosines = numpy.cos(thetas)
    sines = numpy.sin(thetas)
    zetas = numpy.linspace(0.1, 0.5, n_points)
    return numpy.column_stack((cosines, sines, zetas))


def test_polygon_generator_0():

    n_points = 4

    points = circle_inscribed_polygon(n_points)

    faces = [list(range(n_points))]

    v = shapes.ConvexPolygon(points, faces)

    expected = numpy.array([
       [0, 1, 2],
       [0, 2, 3]])

    assert_equal_triangles(v.triangles, expected)


def test_polygon_generator_1():

    n_points = 5

    points = circle_inscribed_polygon(n_points)

    faces = [list(range(n_points))]

    v = shapes.ConvexPolygon(points, faces)

    expected = numpy.array([
       [0, 1, 2],
       [0, 2, 3],
       [0, 3, 4]])

    assert_equal_triangles(v.triangles, expected)


def test_polygon_generator_2():

    n_points = 6

    points = circle_inscribed_polygon(n_points)

    faces = [list(range(n_points))]

    v = shapes.ConvexPolygon(points, faces)

    expected = numpy.array([
       [0, 1, 2],
       [0, 2, 3],
       [0, 3, 4],
       [0, 4, 5]])

    assert_equal_triangles(v.triangles, expected)


def test_polygon_generator_unique_triangulation():
    """ We need to test if the polygon triangulation is always the same between different
    orderings of the vertices. Otherwise we will have face intersection when we consider the triangulation
    in the global index space
    """
    points = circle_inscribed_polygon(6)

    for points in [circle_inscribed_polygon(6), circle_inscribed_polygon(5)]:

        base_array = numpy.arange(len(points), dtype=numpy.int)

        v_ref = shapes.ConvexPolygon(points, [base_array.tolist()])

        for i in range(len(points)):

            faces = [numpy.roll(base_array, i).tolist()]

            print('faces: ', faces)

            v2 = shapes.ConvexPolygon(points, faces)
            assert_equal_triangles(v_ref.triangles, v2.triangles)

        for i in range(len(points)):

            faces = [numpy.roll(base_array[::-1], i).tolist()]

            print('faces: ', faces)

            v2 = shapes.ConvexPolygon(points, faces)
            assert_equal_triangles(v_ref.triangles, v2.triangles)


def test_face_vectors(convex_polygon):

    face_vectors = convex_polygon.face_vectors

    points = convex_polygon.points
    triangles = convex_polygon.triangles

    expected_vectors = (points[triangles[:, 1]] - points[triangles[:, 0]],
                        points[triangles[:, 2]] - points[triangles[:, 0]])

    assert numpy.allclose(face_vectors[0], expected_vectors[0])
    assert numpy.allclose(face_vectors[1], expected_vectors[1])


def test_centroid(convex_polygon):

    expected_centroid = convex_polygon.points.mean(axis=0)

    actual_centroid = convex_polygon.centroid
    assert numpy.allclose(expected_centroid, actual_centroid), '{} : {}'.format(str(expected_centroid),
                                                                           str(actual_centroid))


def test_volume(convex_polygon):

    assert numpy.isclose(convex_polygon.volume,
                         ConvexHull(convex_polygon.points).volume)


def test_inscribed_sphere(convex_polygon):

    center, radius = convex_polygon.inscribed_sphere

    expected_center = convex_polygon._center

    assert numpy.allclose(center, expected_center)

    expected_radius = 0.14433756729740652

    assert numpy.isclose(radius, expected_radius), (radius, expected_radius)


def test_adjacency(convex_polygon):

    adjacency = convex_polygon.adjacency

    expected_adjacency = ({1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2})

    assert adjacency == expected_adjacency, '\n{}\n{}'.format(adjacency, expected_adjacency)
