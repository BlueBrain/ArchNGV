import numpy as np
import pytest
from numpy import testing as npt

from archngv.building.endfeet_reconstruction import area_shrinking as _s


@pytest.fixture
def triangle_areas():
    return np.array(
        [
            84.87747949,
            47.26654929,
            96.57599225,
            4.12815829,
            23.9717335,
            54.11642639,
            49.45504892,
            37.44585053,
            19.26206214,
            15.9478388,
            27.31892394,
            46.33575082,
            78.78258077,
            68.40922939,
            18.70263987,
            34.96866783,
            7.93242081,
            6.30353451,
            90.98777643,
            67.38643288,
        ]
    )


@pytest.fixture
def triangle_travel_times():
    return np.array(
        [
            2.4,
            3.4,
            0.8,
            0.0,
            2.0,
            1.0,
            1.6,
            2.6,
            0.6,
            3.8,
            1.8,
            3.0,
            1.2,
            1.4,
            0.2,
            0.4,
            3.2,
            2.8,
            3.6,
            2.2,
        ]
    )


def test_shrink_surface_mesh(triangle_areas, triangle_travel_times):
    current_area = triangle_areas.sum()
    target_area = current_area * 0.2

    idx = _s.shrink_surface_mesh(triangle_areas, triangle_travel_times, current_area, target_area)

    npt.assert_array_equal(idx, [2, 8, 15, 14, 3])
    assert triangle_areas[idx].sum() <= target_area
