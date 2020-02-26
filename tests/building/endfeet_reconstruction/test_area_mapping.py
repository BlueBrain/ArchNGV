import pytest
import numpy as np
from numpy import testing as npt

from archngv.utils.statistics import truncated_normal
from archngv.building.endfeet_reconstruction import area_mapping as _a


@pytest.fixture
def endfeet_areas():
    return np.array([133.25994113,  379.52550471,  409.87954536,    0.        ,
                     370.3979085 ,  252.92270169,  192.23042575,  440.09876271,
                     1059.00420464,  428.37837456,  356.57605112,    0.      ])


@pytest.fixture
def target_distribution():

    mean = 191.55714285714288
    sdev = 160.55871962923592
    vmax = 442.1

    return truncated_normal(mean, sdev, 0.0, vmax)


def test_sorted_nonzero_areas(endfeet_areas):

    ids, areas = _a._sorted_nonzero_areas(endfeet_areas)

    expected_ids = [0, 6, 5, 10, 4, 1, 2, 9, 7, 8]

    npt.assert_array_equal(expected_ids, ids)
    npt.assert_allclose(endfeet_areas[expected_ids], areas)


def test_map_to_target_distribution(endfeet_areas, target_distribution):

    res = _a.transform_to_target_distribution(endfeet_areas, target_distribution)
    expected_res = np.array([ 55.76502574, 236.81122955, 272.72008571,   0.        ,
                              203.05607038, 135.58730095,  98.58285323, 363.75291088,
                              442.1       , 313.28630898, 169.80703006,   0.        ])

    npt.assert_allclose(expected_res, res)
