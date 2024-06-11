# SPDX-License-Identifier: Apache-2.0

"""
Function potentials that depend on the distance r
"""

import numpy as np


def lenard_jones(dist, d_m, epsilon):
    """Lenard Jones potential to approximate attraction
    repulsion

    Args:
        d_m: float
            Distance at which the potential reaches its min.
        epsilon:
            The depth of the potential well.
    """
    ratio = d_m / dist
    return epsilon * (np.power(ratio, 12) - 2.0 * np.power(ratio, 6))


def coulomb(dist, d_m):
    """Classical Coulomb potential"""
    return d_m / dist**2


def inverse_distance(dist, d_m):
    """One over r repulsion potential"""
    return d_m / dist


def spring(dist, d_m, spring_constant):
    """Spring potential"""
    return spring_constant * (dist - d_m) ** 2
