import numpy as np
from numpy import testing as npt

from archngv.building.endfeet_reconstruction.groups import GroupedElements
from archngv.building.endfeet_reconstruction import area_generation as _a


def test_endfeet_areas():

    triangle_areas = np.random.random(20)

    triangle_ids = np.arange(20, dtype=np.int32)
    np.random.shuffle(triangle_ids)

    group_ids = np.array([-1, 5, 10])

    group_offsets = np.array([0, 5, 10, 20 ])

    g_elements = GroupedElements(triangle_ids, group_offsets, group_ids)

    areas = _a._endfeet_areas(g_elements, triangle_areas, n_endfeet=12)

    expected_areas = np.zeros(12, dtype=np.float32)
    expected_areas[5] = triangle_areas[triangle_ids[group_offsets[1]: group_offsets[2]]].sum()
    expected_areas[10] = triangle_areas[triangle_ids[group_offsets[2]: group_offsets[3]]].sum()


    npt.assert_allclose(expected_areas, areas)


def test_global_to_local_indices():
    
    global_triangles = np.array([
        [6, 7, 8],
        [24, 25, 26],
        [45, 46, 47],
        [42, 43, 44],
        [9, 10, 11]])

    expected_global_vertices = [6, 7, 8, 9, 10, 11, 24, 25, 26, 42, 43, 44, 45, 46, 47]

    expected_local_triangles = [[0, 1, 2],
                                [6, 7, 8],
                                [12, 13, 14],
                                [9, 10, 11],
                                [3, 4, 5]]

    vertices_global, triangles_local = _a._global_to_local_indices(global_triangles)

    npt.assert_array_equal(vertices_global, expected_global_vertices)
    npt.assert_array_equal(triangles_local, expected_local_triangles)


def test_shrink_endfoot_triangles():

    triangles = np.array([
       [ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11],
       [12, 13, 14],
       [15, 16, 17],
       [18, 19, 20],
       [21, 22, 23],
       [24, 25, 26],
       [27, 28, 29],
       [30, 31, 32],
       [33, 34, 35],
       [36, 37, 38],
       [39, 40, 41],
       [42, 43, 44],
       [45, 46, 47],
       [48, 49, 50],
       [51, 52, 53],
       [54, 55, 56],
       [57, 58, 59]])

    triangle_areas = np.array([84.87747949, 47.26654929, 96.57599225,  4.12815829, 23.9717335 ,
                     54.11642639, 49.45504892, 37.44585053, 19.26206214, 15.9478388 ,
                     27.31892394, 46.33575082, 78.78258077, 68.40922939, 18.70263987,
                     34.96866783,  7.93242081,  6.30353451, 90.98777643, 67.38643288])


    triangle_travel_times = np.array([2.4, 3.4, 0.8, 0. , 2. , 1. , 1.6, 2.6,
                                      0.6, 3.8, 1.8, 3. , 1.2, 1.4, 0.2, 0.4, 3.2, 2.8, 3.6, 2.2])

    current_area = triangle_areas.sum()
    target_area = current_area * 0.2

    t_tris = _a._shrink_endfoot_triangles(triangles, triangle_areas, triangle_travel_times, current_area, target_area)
    expected_triangle_ids = np.array([2, 8, 15, 14, 3])
    #expected_vertices = np.unique(triangles[expected_triangle_ids])

    # expected global triangles
    # [[ 6  7  8]
    #  [24 25 26]
    #  [45 46 47]
    #  [42 43 44]
    #  [ 9 10 11]]
    expected_triangles = [[6, 7, 8],
                          [24, 25, 26],
                          [45, 46, 47],
                          [42, 43, 44],
                          [9, 10, 11]]
    """
    expected_local_triangles = [[0, 1, 2],
                                [6, 7, 8],
                                [12, 13, 14],
                                [9, 10, 11],
                                [3, 4, 5]]
    """
    #npt.assert_array_equal(expected_vertices, vertices)
    npt.assert_array_equal(expected_triangles, t_tris)

