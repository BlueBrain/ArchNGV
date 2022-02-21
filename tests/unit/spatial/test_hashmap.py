import numpy as np
import pytest

from archngv.spatial import hashmap


@pytest.fixture(scope="function")
def hashmapbase():

    voxel_size = 12.0
    offset = np.array([10.0, 12.0, 13.0])

    return hashmap.HashMapBase(voxel_size, offset)


@pytest.fixture(scope="function")
def empty_phashmap():

    voxel_size = 30.0
    offset = np.array([50.0, 22.0, 19.0])

    return hashmap.PointHashMap(voxel_size, offset)


@pytest.fixture(scope="function")
def filled_phashmap():

    voxel_size = 15.0
    offset = np.array([10.0, 22.0, 19.0])

    points = [
        (61.0, 330.0, 450.0),
        (98.0, 220.0, 200.0),
        (767.0, 430.0, 1000.0),
        (520.0, 111.0, 433.0),
        (9123.0, 2323.0, 1232.0),
        (390.0, 200.0, 145.0),
        (470.0, 200.0, 145.0),
        (480.0, 200.0, 145.0),
    ]

    hm = hashmap.PointHashMap(voxel_size, offset)

    for point in points:
        hm.add_point(point)

    return hm


def test_key(hashmapbase):

    point = np.array([32.0, 44.0, 13.0])

    i, j, k = hashmapbase.key(point)

    assert i == 1 and j == 2 and k == 0


def test_add_point(empty_phashmap):
    def _check(t1, t2):
        return t1[0] == t2[0] and t1[1] == t2[1] and t1[2] == t2[2]

    points = [(61.0, 330.0, 450.0), (98.0, 220.0, 200.0), (767.0, 430.0, 1000.0)]

    for point in points:
        empty_phashmap.add_point(point)

    keys = [(0, 10, 14), (1, 6, 6), (23, 13, 32)]

    for i in range(3):
        assert _check(empty_phashmap._d[keys[i]][0], points[i]), (
            empty_phashmap._d[keys[i]],
            points[i],
        )


def test_gen(filled_phashmap):

    # [(50, 27, 65), (3, 20, 28), (25, 11, 8), (5, 13, 12), (34, 5, 27), (607, 153, 80)]

    ijk_min = (2, 2, 2)
    ijk_max = (30, 30, 30)

    points = list(filled_phashmap._gen(ijk_min, ijk_max))

    assert (61.0, 330.0, 450.0) in points
    assert (98.0, 220.0, 200.0) in points
    assert (390.0, 200.0, 145.0) in points
    assert (470.0, 200.0, 145.0) in points

    assert len(points) == 4
