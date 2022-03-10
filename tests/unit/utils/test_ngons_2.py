import numpy as np
import pytest

from archngv.utils import ngons as _impl


@pytest.fixture
def sorted_points():
    return np.array(
        [
            [742.55783358, 699.57242297, 751.95820905],
            [742.89264973, 704.62878605, 749.44423881],
            [744.78617453, 688.85998334, 724.63880534],
            [745.9658508, 697.55238907, 754.71250494],
            [746.49159632, 712.24310942, 711.26753817],
            [746.75512463, 695.46575521, 704.33502008],
            [753.87133647, 677.86409645, 714.58393513],
            [754.9214507, 702.27071703, 756.63710379],
            [755.83155481, 682.25475179, 698.77759204],
            [756.69741549, 677.98758544, 704.31038325],
            [764.99854104, 724.31983794, 749.9321204],
            [765.30683236, 723.29886971, 750.6254617],
            [773.60114476, 662.37187914, 718.75359183],
            [780.41184939, 741.20895435, 722.85445005],
            [781.09176065, 741.27440136, 727.55710532],
            [786.49803671, 742.85868789, 723.4682142],
            [792.40263814, 740.48642534, 722.42278167],
            [795.23111357, 673.22526692, 694.61031859],
            [796.11730782, 671.14081224, 697.1172658],
            [799.63174946, 644.55397891, 732.38423286],
            [802.8989143, 705.63087297, 746.1079414],
            [807.4252622, 642.93675448, 747.99169308],
            [810.58614559, 661.72230421, 752.59281508],
            [811.97491549, 638.04313499, 744.85677727],
            [811.98541605, 645.77183227, 749.07077193],
            [812.06661714, 638.06503605, 744.85932831],
        ]
    )


@pytest.fixture
def triangles1():
    return np.array(
        [
            [22, 17, 1],
            [23, 22, 1],
            [2, 3, 1],
            [5, 2, 1],
            [8, 5, 1],
            [9, 8, 1],
            [17, 9, 1],
            [13, 23, 1],
            [21, 13, 1],
            [3, 21, 1],
            [21, 3, 2],
            [20, 21, 2],
            [10, 20, 2],
            [25, 10, 2],
            [18, 25, 2],
            [5, 18, 2],
            [8, 9, 4],
            [0, 8, 4],
            [11, 0, 4],
            [7, 11, 4],
            [17, 22, 4],
            [9, 17, 4],
            [6, 7, 4],
            [14, 6, 4],
            [16, 14, 4],
            [13, 16, 4],
            [23, 13, 4],
            [22, 23, 4],
            [24, 18, 5],
            [12, 24, 5],
            [15, 12, 5],
            [0, 15, 5],
            [8, 0, 5],
            [0, 11, 6],
            [15, 0, 6],
            [14, 15, 6],
            [11, 7, 6],
            [21, 20, 10],
            [13, 21, 10],
            [16, 13, 10],
            [19, 16, 10],
            [12, 19, 10],
            [24, 12, 10],
            [25, 24, 10],
            [16, 19, 12],
            [14, 16, 12],
            [15, 14, 12],
            [24, 25, 18],
        ],
        dtype=np.uint64,
    )


def test_local_to_global_mapping__identity(sorted_points, triangles1):

    offsets = (
        np.array([0, 0], dtype=np.uint64),
        np.array([len(sorted_points), len(triangles1)], dtype=np.uint64),
    )

    res_points, res_triangles = _impl.local_to_global_mapping(sorted_points, triangles1, offsets)

    np.testing.assert_allclose(res_points, sorted_points)
    np.testing.assert_allclose(res_triangles, triangles1)


@pytest.fixture
def domain_polygon_triangles():
    triangles = np.array(
        [
            [21, 25, 3],
            [21, 3, 28],
            [21, 28, 15],
            [21, 15, 23],
            [21, 23, 1],
            [21, 17, 0],
            [21, 0, 1],
            [1, 0, 14],
            [1, 14, 23],
            [18, 2, 11],
            [18, 11, 6],
            [18, 6, 9],
            [18, 9, 12],
            [18, 12, 24],
            [17, 0, 14],
            [17, 14, 19],
            [17, 19, 18],
            [17, 18, 2],
            [21, 25, 10],
            [21, 10, 11],
            [21, 11, 2],
            [21, 2, 17],
            [3, 29, 28],
        ],
        dtype=np.uint64,
    )

    points = np.array(
        [
            [0.71407298, 0.18955103, 0.70725155],
            [0.52831547, 0.55953882, 0.21106774],
            [0.78351118, 0.44071989, 0.53574945],
            [0.94859445, 0.088215, 0.31478145],
            [0.32258222, 0.46279019, 0.62868786],
            [0.81029217, 0.28015308, 0.90365601],
            [0.95217861, 0.32080787, 0.52296476],
            [0.52620329, 0.29123142, 0.58001346],
            [0.25875277, 0.16994707, 0.27976299],
            [0.27580844, 0.64500242, 0.9313845],
            [0.11948365, 0.24748816, 0.6379533],
            [0.29980992, 0.93001245, 0.56379982],
            [0.3392728, 0.27146855, 0.80186385],
            [0.70786638, 0.21692713, 0.55257139],
            [0.42733557, 0.42351681, 0.29217951],
            [0.13903866, 0.83717517, 0.41839472],
            [0.44684757, 0.56135018, 0.75034792],
            [0.3962944, 0.24219984, 0.83485044],
            [0.72403478, 0.13071367, 0.46410423],
            [0.68512179, 0.17835754, 0.19348947],
            [0.80949961, 0.8743832, 0.22824881],
            [0.74114653, 0.58954571, 0.81840249],
            [0.79115557, 0.60977578, 0.69864708],
            [0.2070119, 0.87583752, 0.75028292],
            [0.75307522, 0.27093624, 0.97143941],
            [0.19032416, 0.19866327, 0.83348639],
            [0.53684528, 0.22214462, 0.03445889],
            [0.21538014, 0.50500167, 0.87710681],
            [0.03337195, 0.20005121, 0.6138431],
            [0.94572668, 0.50176995, 0.98838023],
        ]
    )

    polygon_ids = np.array(
        [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6], dtype=np.uint64
    )

    ps_tris_offsets = np.array([(0, 0), (len(points), len(triangles))], dtype=np.uint64)
    return points, triangles, polygon_ids, ps_tris_offsets


@pytest.fixture
def domain_neighbor1():
    points = np.array(
        [
            [755.83154, 682.25476, 698.7776],
            [811.9854, 645.77185, 749.0708],
            [802.8989, 705.63086, 746.108],
            [810.5861, 661.7223, 752.59283],
            [799.6318, 644.55396, 732.3842],
            [792.40265, 740.48645, 722.4228],
            [753.87134, 677.8641, 714.5839],
            [773.60114, 662.3719, 718.7536],
            [795.23114, 673.2253, 694.6103],
            [796.1173, 671.1408, 697.11725],
            [764.99854, 724.3198, 749.9321],
            [756.6974, 677.9876, 704.31036],
            [746.4916, 712.2431, 711.2675],
            [745.9658, 697.55237, 754.7125],
            [744.7862, 688.86, 724.6388],
            [746.7551, 695.46576, 704.335],
            [742.55786, 699.57245, 751.9582],
            [812.0666, 638.06506, 744.8593],
            [786.49805, 742.8587, 723.4682],
            [742.89264, 704.6288, 749.4442],
            [765.3068, 723.2989, 750.6255],
            [754.92145, 702.2707, 756.6371],
            [811.9749, 638.04315, 744.85675],
            [807.42523, 642.93677, 747.9917],
            [780.41187, 741.209, 722.85443],
            [781.09174, 741.2744, 727.5571],
        ],
        dtype=np.float32,
    )

    triangles = np.array(
        [
            [22, 17, 1],
            [22, 1, 23],
            [8, 9, 17],
            [8, 17, 1],
            [8, 1, 3],
            [8, 3, 2],
            [8, 2, 5],
            [23, 1, 3],
            [23, 3, 21],
            [23, 21, 13],
            [2, 20, 21],
            [2, 21, 3],
            [5, 18, 25],
            [5, 25, 10],
            [5, 10, 20],
            [5, 20, 2],
            [8, 9, 4],
            [8, 4, 7],
            [8, 7, 11],
            [8, 11, 0],
            [9, 17, 22],
            [9, 22, 4],
            [6, 7, 4],
            [6, 4, 22],
            [6, 22, 23],
            [6, 23, 13],
            [6, 13, 16],
            [6, 16, 14],
            [8, 0, 15],
            [8, 15, 12],
            [8, 12, 24],
            [8, 24, 18],
            [8, 18, 5],
            [0, 11, 6],
            [0, 6, 14],
            [0, 14, 15],
            [6, 7, 11],
            [19, 16, 13],
            [19, 13, 21],
            [19, 21, 20],
            [19, 20, 10],
            [12, 24, 25],
            [12, 25, 10],
            [12, 10, 19],
            [15, 14, 16],
            [15, 16, 19],
            [15, 19, 12],
            [18, 25, 24],
        ],
        dtype=np.uint64,
    )

    polygon_ids = np.array(
        [
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            3,
            3,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            6,
            6,
            7,
            7,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            8,
            9,
            9,
            9,
            10,
            11,
            11,
            11,
            11,
            12,
            12,
            12,
            13,
            13,
            13,
            14,
        ],
        dtype=np.uint64,
    )

    return points, triangles, polygon_ids


@pytest.fixture
def domain_neighbor2():

    triangles = np.array(
        [
            [1, 15, 5],
            [1, 5, 12],
            [1, 12, 13],
            [1, 13, 8],
            [1, 8, 20],
            [1, 20, 9],
            [1, 15, 14],
            [1, 14, 0],
            [1, 0, 6],
            [1, 6, 2],
            [1, 2, 9],
            [9, 2, 21],
            [9, 21, 20],
            [6, 0, 16],
            [6, 16, 7],
            [6, 7, 3],
            [6, 3, 21],
            [6, 21, 2],
            [20, 21, 3],
            [20, 3, 8],
            [13, 8, 3],
            [13, 3, 7],
            [13, 7, 18],
            [13, 18, 11],
            [10, 5, 12],
            [10, 12, 4],
            [12, 13, 11],
            [12, 11, 17],
            [12, 17, 4],
            [0, 14, 10],
            [0, 10, 4],
            [0, 4, 17],
            [0, 17, 19],
            [0, 19, 16],
            [15, 14, 10],
            [15, 10, 5],
            [7, 16, 19],
            [7, 19, 18],
            [11, 17, 19],
            [11, 19, 18],
        ],
        dtype=np.uint64,
    )

    polygon_ids = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            8,
            8,
            8,
            9,
            9,
            9,
            9,
            9,
            10,
            10,
            11,
            11,
            12,
            12,
        ],
        dtype=np.uint64,
    )

    points = np.array(
        [
            [724.9809, 635.0, 723.9804],
            [753.87134, 677.8641, 714.5839],
            [737.1956, 684.51843, 727.89966],
            [741.3693, 695.2172, 758.6605],
            [811.6408, 635.0, 744.96625],
            [799.6318, 644.55396, 732.3842],
            [728.4798, 650.1613, 720.4845],
            [735.132, 665.3451, 778.34875],
            [745.9658, 697.55237, 754.7125],
            [744.7862, 688.86, 724.6388],
            [796.6898, 635.0, 730.95374],
            [752.2637, 636.7071, 783.6743],
            [811.9749, 638.04315, 744.85675],
            [807.42523, 642.93677, 747.9917],
            [786.2423, 635.0, 725.55096],
            [773.60114, 662.3719, 718.7536],
            [727.9164, 635.0, 781.20215],
            [752.1512, 635.0, 783.68695],
            [735.9016, 646.77454, 787.8639],
            [732.8084, 635.0, 788.53156],
            [742.55786, 699.57245, 751.9582],
            [742.0483, 699.35315, 752.3359],
        ],
        dtype=np.float32,
    )

    return points, triangles, polygon_ids


@pytest.fixture
def neighboring_domains_tessellation(domain_neighbor1, domain_neighbor2):
    points1, triangles1, polygon_ids1 = domain_neighbor1
    points2, triangles2, polygon_ids2 = domain_neighbor2

    ps_tris_offsets = np.array(
        [
            (0, 0),
            (len(points1), len(triangles1)),
            (len(points1) + len(points2), len(triangles1) + len(triangles2)),
        ],
        dtype=np.uint64,
    )

    return (
        np.vstack((points1, points2)),
        np.vstack((triangles1, triangles2)),
        np.hstack((polygon_ids1, polygon_ids2)),
        ps_tris_offsets,
    )


def test_local_to_global_polygon_ids(domain_polygon_triangles):

    _, triangles, polygon_ids, _ = domain_polygon_triangles

    polygon_ids = np.array(
        [
            0,
            0,
            1,
            1,
            2,
            0,
            0,
            0,
            1,
            1,
            2,
            2,
            0,
            0,
            1,
            2,
            2,
            3,
            3,
            4,
            5,
            6,
            0,
            1,
            2,
            2,
            3,
            4,
            5,
            5,
            6,
        ]
    )

    offsets = np.array([0, 5, 12, 22, 31])

    global_polygon_ids = _impl.local_to_global_polygon_ids(polygon_ids)

    expected_ids = np.array(
        [
            0,
            0,
            1,
            1,
            2,
            3,
            3,
            3,
            4,
            4,
            5,
            5,
            6,
            6,
            7,
            8,
            8,
            9,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            15,
            16,
            17,
            18,
            18,
            19,
        ]
    )

    np.testing.assert_allclose(global_polygon_ids, expected_ids)


def test_triangles_to_polygons__one_domain_local(domain_polygon_triangles):

    _, triangles, polygon_ids, _ = domain_polygon_triangles

    polygons = _impl.triangles_to_polygons(triangles, polygon_ids)

    expected_polygons = [
        [21, 25, 3, 28, 15, 23, 1],
        [21, 17, 0, 1],
        [1, 0, 14, 23],
        [18, 2, 11, 6, 9, 12, 24],
        [17, 0, 14, 19, 18, 2],
        [21, 25, 10, 11, 2, 17],
        [3, 29, 28],
    ]

    assert polygons == expected_polygons


def test_triangles_to_polygons__one_domain_global(domain_polygon_triangles):

    local_ps, local_tris, local_polys, ps_tris_offsets = domain_polygon_triangles

    global_ps, global_tris, global_polys = _impl.local_to_global_mapping(
        local_ps,
        local_tris,
        ps_tris_offsets,
        triangle_labels=_impl.local_to_global_polygon_ids(local_polys),
    )

    polygons = _impl.triangles_to_polygons(global_tris, global_polys)

    expected_polygons = [
        [21, 25, 3, 28, 15, 23, 1],
        [21, 17, 0, 1],
        [1, 0, 14, 23],
        [18, 2, 11, 6, 9, 12, 24],
        [17, 0, 14, 19, 18, 2],
        [21, 25, 10, 11, 2, 17],
        [3, 29, 28],
    ]

    assert polygons == expected_polygons


@pytest.fixture
def domains_one_polygon():

    triangles = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [5, 6, 7]])

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 5.0],
            [6.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ]
    )

    polygon_ids = np.array([0, 0, 0, 0, 1])

    ps_tris_offsets = np.array(
        [[0, 0], [len(points), len(triangles)], [2 * len(points), 2 * len(triangles)]]
    )

    return (
        np.vstack([points, points]),
        np.vstack([triangles, triangles]),
        np.hstack([polygon_ids, polygon_ids]),
        ps_tris_offsets,
    )


def test_triangles_to_polygons__two_neighboring_domains(domains_one_polygon):

    ps, tris, poly_ids, ps_tris_offsets = domains_one_polygon

    g_poly_ids = _impl.local_to_global_polygon_ids(poly_ids)
    ps, tris, polys = _impl.local_to_global_mapping(
        ps, tris, ps_tris_offsets, triangle_labels=g_poly_ids
    )

    polygons = _impl.triangles_to_polygons(tris, polys)

    assert polygons == [[0, 1, 2, 3, 4, 5], [5, 6, 7]]


"""
def test_triangles_to_polygons__two_neighboring_domains(neighboring_domains_tessellation):

    ps, tris, poly_ids, ps_tris_offsets = neighboring_domains_tessellation

    g_poly_ids = _impl.local_to_global_polygon_ids(poly_ids)
    ps, tris, polys = _impl.local_to_global_mapping(ps, tris, ps_tris_offsets, triangle_labels=g_poly_ids)

    polygons = _impl.triangles_to_polygons(tris, polys)

    assert False"""
