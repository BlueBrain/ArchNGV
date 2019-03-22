"""
Function potentials that depend on the distance r
"""

import numpy as np


def lenard_jones(r, rm, e):
    """ Lenard Jones potential to approximate attraction
    repulsion

    rm: distance at which the potential reaches its min
    e: the depth of the potential well
    """
    ratio = rm / r
    return e * (np.power(ratio, 12) - 2. * np.power(ratio, 6))


def coulomb(r, rm):
    """ Classical Coulomb potential
    """
    return rm / r ** 2


def inverse_distance(r, rm):
    """ One over r repulsion potential
    """
    return rm / r

def spring(r, d, k):
    """ Spring potential
    """
    return k * (r - d) ** 2
