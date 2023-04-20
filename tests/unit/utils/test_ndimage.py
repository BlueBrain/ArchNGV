import numpy
import numpy.testing
import pandas.testing
import pytest
import scipy.ndimage
import voxcell

from archngv.utils import ndimage as test_module

SHAPE = (10, 10, 10)


@pytest.fixture
def volume_empty():
    return numpy.zeros(shape=SHAPE, dtype=int)


@pytest.fixture
def volume_one_component():
    arr = numpy.zeros(shape=SHAPE, dtype=int)

    arr[4:8, 2:3, 8:9] = 1

    return arr


@pytest.fixture
def volume_two_components():
    arr = numpy.zeros(shape=SHAPE, dtype=int)

    arr[2:4, 2:4, 0:9] = 1
    arr[6:8, 6:8, 0:9] = 1

    return arr


@pytest.fixture
def volume_two_components_different_value():
    arr = numpy.zeros(shape=SHAPE, dtype=int)
    values = (1, 2)
    arr[2:4, 2:4, 0:9] = values[0]
    arr[6:8, 6:8, 0:9] = values[1]

    return arr, values


def test_connected_components__empty(volume_empty):
    result, n_components = test_module.connected_components(volume_empty)

    assert numpy.all(result == 0)
    assert n_components == 0


def test_connected_components__one_components(volume_one_component):
    result, n_components = test_module.connected_components(volume_one_component)

    assert n_components == 1
    assert numpy.all(result[4:8, 2:3, 8:9] == 1)


def test_connected_components__two_components(volume_two_components):
    result, n_components = test_module.connected_components(volume_two_components)

    assert n_components == 2
    assert numpy.all(result[2:4, 2:4, 0:9] == 1)
    assert numpy.all(result[6:8, 6:8, 0:9] == 2)


def test_connected_components_sub_array(volume_two_components_different_value):
    data = volume_two_components_different_value[0]
    values = volume_two_components_different_value[1]
    for sub_array, value in zip(test_module.connected_components_sub_array(data), values):
        assert sub_array.max() == value


def test_map_positions_to_connected_components():
    region_mask_raw = numpy.zeros((5, 5, 5), dtype=bool)
    region_mask_raw[0][0][0] = True
    region_mask_raw[0][4][0] = True
    region_mask = voxcell.VoxelData(region_mask_raw, voxel_dimensions=[1, 1, 1])

    positions = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    nb_pixel_component_threshold = 1
    (bb1, ids1), (bb2, ids2) = list(
        test_module.map_positions_to_connected_components(
            positions, region_mask, nb_pixel_component_threshold
        )
    )

    numpy.testing.assert_array_equal(bb1.min_point, [0.0, 0.0, 0.0])
    numpy.testing.assert_array_equal(bb1.max_point, [1.0, 1.0, 1.0])

    numpy.testing.assert_array_equal(bb2.min_point, [0.0, 4.0, 0.0])
    numpy.testing.assert_array_equal(bb2.max_point, [1.0, 5.0, 1.0])

    numpy.testing.assert_array_equal(ids1, [0, 2])
    numpy.testing.assert_array_equal(ids2, [1])

    # Test the threshold
    region_mask_raw = numpy.zeros((5, 5, 5), dtype=bool)
    region_mask_raw[0][0][0] = True
    region_mask_raw[0][1][0] = True
    region_mask_raw[1][0][0] = True
    region_mask_raw[1][1][0] = True

    region_mask_raw[0][0][3] = True
    region_mask = voxcell.VoxelData(region_mask_raw, voxel_dimensions=[1, 1, 1])

    positions = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 3.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    nb_pixel_component_threshold = 3

    ((bb1, ids1),) = list(
        test_module.map_positions_to_connected_components(
            positions, region_mask, nb_pixel_component_threshold
        )
    )
    numpy.testing.assert_array_equal(bb1.min_point, [0.0, 0.0, 0.0])
    numpy.testing.assert_array_equal(bb1.max_point, [2.0, 2.0, 1.0])
    numpy.testing.assert_array_equal(ids1, [0, 1, 3])


def test_map_positions_to_connected_components_negative_dimention():
    region_mask_raw = numpy.zeros((5, 5, 5), dtype=bool)
    region_mask_raw[0][0][0] = True
    region_mask_raw[0][4][0] = True
    region_mask = voxcell.VoxelData(region_mask_raw, voxel_dimensions=[1, -1, 1])

    positions = numpy.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, -4.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    nb_pixel_component_threshold = 1
    (bb1, ids1), (bb2, ids2) = list(
        test_module.map_positions_to_connected_components(
            positions, region_mask, nb_pixel_component_threshold
        )
    )

    numpy.testing.assert_array_equal(bb1.min_point, [0.0, -1.0, 0.0])
    numpy.testing.assert_array_equal(bb1.max_point, [1.0, 0.0, 1.0])

    numpy.testing.assert_array_equal(bb2.min_point, [0.0, -5.0, 0.0])
    numpy.testing.assert_array_equal(bb2.max_point, [1.0, -4.0, 1.0])

    numpy.testing.assert_array_equal(ids1, [0, 2])
    numpy.testing.assert_array_equal(ids2, [1])
