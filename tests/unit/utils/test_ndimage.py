import numpy
import numpy.testing
import pytest
import scipy.ndimage

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
