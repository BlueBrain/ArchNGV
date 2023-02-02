import numpy as np
import pytest

from archngv.utils import ngons


def test_vectorized_consecutive_triangle_vectors():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    triangles = np.array([[3, 1, 0], [2, 1, 0], [2, 3, 0], [2, 3, 1]])

    face_vectors = ngons.vectorized_consecutive_triangle_vectors(points, triangles)

    expected_vectors = (
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )

    assert np.allclose(face_vectors[0], expected_vectors[0])
    assert np.allclose(face_vectors[1], expected_vectors[1])


def test_vectorized_triangle_normal():
    As = np.random.rand(2, 3)
    Bs = np.random.rand(2, 3)
    Cs = np.random.rand(2, 3)

    Us = Bs - As
    Vs = Cs - As

    result = []

    for i in range(2):
        x = (Us[i][1] * Vs[i][2]) - (Us[i][2] * Vs[i][1])
        y = (Us[i][2] * Vs[i][0]) - (Us[i][0] * Vs[i][2])
        z = (Us[i][0] * Vs[i][1]) - (Us[i][1] * Vs[i][0])
        dist = np.sqrt(x**2 + y**2 + z**2)
        n = np.asarray((x / dist, y / dist, z / dist))
        result.append(n)

    expected_result = np.asarray(result)

    result = ngons.vectorized_triangle_normal(Bs - As, Cs - As)

    assert np.allclose(expected_result, result)


def test_vectorized_parallelepiped_volume():
    vectors1 = np.array([(1.0, 1.0, 1.0), (5.0, 1.0, 3.0)])
    vectors2 = np.array([(2.0, 1.0, 2.0), (4.0, 1.0, 2.0)])
    vectors3 = np.array([(2.0, 4.0, 4.0), (2.0, 1.0, 4.0)])

    res_arr = ngons.vectorized_parallelepiped_volume(vectors1, vectors2, vectors3)
    act_arr = np.array([2.0, 4.0])

    assert np.allclose(res_arr, act_arr)


def test_vectorized_tetrahedron_volume():
    vectors1 = np.array([(1.0, 1.0, 1.0), (5.0, 1.0, 3.0)])
    vectors2 = np.array([(2.0, 1.0, 2.0), (4.0, 1.0, 2.0)])
    vectors3 = np.array([(2.0, 4.0, 4.0), (2.0, 1.0, 4.0)])

    res_arr = ngons.vectorized_tetrahedron_volume(vectors1, vectors2, vectors3)
    act_arr = np.array([2.0, 4.0]) / 6.0

    assert np.allclose(res_arr, act_arr)


def test_subdivide_triangles_iterations_0():
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]])
    triangles = [[0, 1, 2]]

    added_points, res_triangles = ngons.subdivide_triangles(points, triangles)

    assert len(added_points) == 1
    assert len(res_triangles) == 3

    assert np.allclose([[1.0, 0.0, 1.0]], added_points)
    assert np.array_equal(res_triangles, [[0, 3, 1], [1, 3, 2], [2, 3, 0]])


def test_subdivide_triangles_iterations_1():
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 9.0], [9.0, 0.0, 0.0]])
    triangles = [[0, 1, 2]]

    added_points, res_triangles = ngons.subdivide_triangles(points, triangles, max_level=1)

    assert len(added_points) == 4
    assert len(res_triangles) == 9

    expected_points = np.array([[3.0, 0.0, 3.0], [1.0, 0.0, 4.0], [4.0, 0.0, 4.0], [4.0, 0.0, 1.0]])
    expected_triangles = np.array(
        [
            [0, 4, 3],
            [3, 4, 1],
            [1, 4, 0],
            [1, 5, 3],
            [3, 5, 2],
            [2, 5, 1],
            [2, 6, 3],
            [3, 6, 0],
            [0, 6, 2],
        ]
    )

    assert np.allclose(expected_points, added_points)
    assert np.array_equal(expected_triangles, res_triangles)


def test_subdivide_triangles_by_total_area__no_subdivision():
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 9.0], [9.0, 0.0, 0.0]])
    triangles = [[0, 1, 2]]

    result_points, result_triangles = ngons.subdivide_triangles_by_total_area(points, triangles, 0)

    assert np.allclose(result_points, points)
    assert np.array_equal(triangles, result_triangles)

    result_points, result_triangles = ngons.subdivide_triangles_by_total_area(points, triangles, 1)

    assert np.allclose(result_points, points)
    assert np.array_equal(triangles, result_triangles)

    result_points, result_triangles = ngons.subdivide_triangles_by_total_area(points, triangles, 2)

    assert np.allclose(result_points, points)
    assert np.array_equal(triangles, result_triangles)

    result_points, result_triangles = ngons.subdivide_triangles_by_total_area(points, triangles, 3)

    print(result_points, result_triangles)

    assert np.allclose(result_points, points)
    assert np.array_equal(triangles, result_triangles)


def test_subdivide_triangles_by_total_area_1():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0],
            [9.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1],
            [0.1, 0.0, 0.0],
        ]
    )
    triangles = [[0, 1, 2], [3, 4, 5]]

    result_points, result_triangles = ngons.subdivide_triangles_by_total_area(points, triangles, 7)

    expected_points = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 9.0],
        [9.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1],
        [0.1, 0.0, 0.0],
        [3.0, 0.0, 3.0],
    ]

    expected_triangles = [[0, 6, 1], [1, 6, 2], [2, 6, 0], [3, 4, 5]]

    print(result_points, expected_points)

    assert np.allclose(result_points, expected_points)
    assert np.array_equal(result_triangles, expected_triangles)


def test_subdivide_triangles_by_total_area_2():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0],
            [9.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 8.0],
            [8.0, 0.0, 0.0],
        ]
    )
    triangles = [[0, 1, 2], [3, 4, 5]]

    result_points, result_triangles = ngons.subdivide_triangles_by_total_area(points, triangles, 30)

    expected_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 9.0],
            [9.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 8.0],
            [8.0, 0.0, 0.0],
            [3.0, 0.0, 3.0],
            [1.0, 0.0, 4.0],
            [4.0, 0.0, 4.0],
            [4.0, 0.0, 1.0],
            [1.33333333, 0.0, 2.33333333],
            [1.33333333, 0.0, 5.33333333],
            [0.33333333, 0.0, 4.33333333],
            [2.33333333, 0.0, 5.33333333],
            [5.33333333, 0.0, 2.33333333],
            [4.33333333, 0.0, 4.33333333],
            [5.33333333, 0.0, 1.33333333],
            [2.33333333, 0.0, 1.33333333],
            [4.33333333, 0.0, 0.33333333],
            [2.66666667, 0.0, 2.66666667],
            [0.88888889, 0.0, 3.55555556],
            [3.55555556, 0.0, 3.55555556],
            [3.55555556, 0.0, 0.88888889],
            [1.18518519, 0.0, 2.07407407],
            [1.18518519, 0.0, 4.74074074],
            [0.2962963, 0.0, 3.85185185],
            [2.07407407, 0.0, 4.74074074],
            [4.74074074, 0.0, 2.07407407],
            [3.85185185, 0.0, 3.85185185],
            [4.74074074, 0.0, 1.18518519],
        ]
    )

    expected_triangles = np.array(
        [
            [0, 10, 7],
            [7, 10, 6],
            [6, 10, 0],
            [6, 11, 7],
            [7, 11, 1],
            [1, 11, 6],
            [1, 12, 7],
            [7, 12, 0],
            [0, 12, 1],
            [1, 13, 8],
            [8, 13, 6],
            [6, 13, 1],
            [6, 14, 8],
            [8, 14, 2],
            [2, 14, 6],
            [2, 15, 8],
            [8, 15, 1],
            [1, 15, 2],
            [2, 16, 9],
            [9, 16, 6],
            [6, 16, 2],
            [6, 17, 9],
            [9, 17, 0],
            [0, 17, 6],
            [0, 18, 9],
            [9, 18, 2],
            [2, 18, 0],
            [3, 23, 20],
            [20, 23, 19],
            [19, 23, 3],
            [19, 24, 20],
            [20, 24, 4],
            [4, 24, 19],
            [4, 25, 20],
            [20, 25, 3],
            [3, 25, 4],
            [4, 26, 21],
            [21, 26, 19],
            [19, 26, 4],
            [19, 27, 21],
            [21, 27, 5],
            [5, 27, 19],
            [5, 28, 21],
            [21, 28, 4],
            [4, 28, 5],
            [5, 29, 22],
            [22, 29, 19],
            [19, 29, 5],
        ]
    )

    assert np.allclose(result_points, expected_points)
    assert np.array_equal(result_triangles, expected_triangles)


def assert_equal_triangles(tris1, tris2):
    tris1 = np.sort(tris1, axis=1)
    tris2 = np.sort(tris2, axis=1)

    tris1 = tris1[np.lexsort(tris1.T)]
    tris2 = tris2[np.lexsort(tris2.T)]

    print(tris1)
    print(tris2)

    np.testing.assert_allclose(tris1, tris2)


def circle_inscribed_polygon(n_points):
    thetas = np.linspace(0.0, 1.8 * np.pi, n_points)
    cosines = np.cos(thetas)
    sines = np.sin(thetas)
    zetas = np.linspace(0.1, 0.5, n_points)
    return np.column_stack((cosines, sines, zetas))


def test_polygons_to_triangles_0():
    n_points = 4

    points = circle_inscribed_polygon(n_points)

    faces = [list(range(n_points))]

    tris, tris_to_polys_map = ngons.polygons_to_triangles(points, faces)

    expected = np.array([[0, 1, 2], [0, 2, 3]])

    assert_equal_triangles(tris, expected)
    np.testing.assert_allclose(tris_to_polys_map, np.zeros(len(expected), dtype=np.uintp))


def test_polygon_to_triangles_1():
    n_points = 5

    points = circle_inscribed_polygon(n_points)

    faces = [list(range(n_points))]

    tris, tris_to_polys_map = ngons.polygons_to_triangles(points, faces)

    expected = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])

    assert_equal_triangles(tris, expected)
    np.testing.assert_allclose(tris_to_polys_map, np.zeros(len(expected), dtype=np.uintp))


def test_polygon_to_triangles_2():
    n_points = 6

    points = circle_inscribed_polygon(n_points)

    faces = [list(range(n_points))]

    tris, tris_to_polys_map = ngons.polygons_to_triangles(points, faces)

    expected = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]])

    assert_equal_triangles(tris, expected)
    np.testing.assert_allclose(tris_to_polys_map, np.zeros(len(expected), dtype=np.uintp))


def test_polygon_to_triangles__unique_triangulation():
    """We need to test if the polygon triangulation is always the same between different
    orderings of the vertices. Otherwise we will have face intersection when we consider the triangulation
    in the global index space
    """
    points = circle_inscribed_polygon(6)

    for points in [circle_inscribed_polygon(6), circle_inscribed_polygon(5)]:
        base_array = np.arange(len(points), dtype=np.int32)

        ref_tris, ref_tris_to_polys_map = ngons.polygons_to_triangles(points, [base_array])

        for i in range(len(points)):
            faces = [np.roll(base_array, i).tolist()]

            print("faces: ", faces)

            tris, tris_to_polys_map = ngons.polygons_to_triangles(points, faces)
            assert_equal_triangles(ref_tris, tris)
            np.testing.assert_allclose(tris_to_polys_map, ref_tris_to_polys_map)

        for i in range(len(points)):
            faces = [np.roll(base_array[::-1], i).tolist()]

            print("faces: ", faces)

            tris, tris_to_polys_map = ngons.polygons_to_triangles(points, faces)
            assert_equal_triangles(ref_tris, tris)
            np.testing.assert_allclose(tris_to_polys_map, ref_tris_to_polys_map)
