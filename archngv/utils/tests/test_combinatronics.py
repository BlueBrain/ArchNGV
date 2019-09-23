import numpy as np
from .. import combinatronics as cb


def test_cartesian_product():

    arr1 = np.array([4, 8])
    arr2 = np.array([1, 2])
    arr3 = np.array([0, 3])

    expected_result = \
        np.array([[4, 1, 0],
                     [4, 1, 3],
                     [4, 2, 0],
                     [4, 2, 3],
                     [8, 1, 0],
                     [8, 1, 3],
                     [8, 2, 0],
                     [8, 2, 3]], dtype=np.intp)

    result = cb.cartesian_product(arr1, arr2, arr3)

    assert np.allclose(expected_result, result)


