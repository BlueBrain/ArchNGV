# SPDX-License-Identifier: Apache-2.0

""" Functions related to combinatronics
"""
import numpy as np


def cartesian_product(arr1, arr2, arr3):
    """Returns the unique combinations of the three arrays"""
    i_x = np.indices((arr1.size, arr2.size, arr3.size), dtype=np.intp)
    i_x = i_x.reshape(3, -1).T

    i_x[:, 0] = arr1[i_x[:, 0]]
    i_x[:, 1] = arr2[i_x[:, 1]]
    i_x[:, 2] = arr3[i_x[:, 2]]

    return i_x
