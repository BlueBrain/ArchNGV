# SPDX-License-Identifier: Apache-2.0

""" Functional helper functions
"""

import itertools


def consecutive_pairs(iterable):
    """It dispatches an iterable in
    consecutive pairs.

    Example:
        iterable = [2, 3, 1, 5, 6]
        result = [[2, 3], [3, 1], [1, 5], [5, 6]]
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
