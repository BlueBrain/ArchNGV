""" Overalap for microdomain geometry """

import logging

import numpy as np

L = logging.getLogger("__name__")


def convex_polygon_with_overlap(centroid, points, overlap_factor):
    """Given the centroid of a convex polygon and its points,
    uniformly dilate in order to expand by the overlap factor. However
    the neighbors inflate as well. Thus, the result overlap between the cell
    and the union of neighbors will be:

    a = (Vinflated - Vdeflated) / Vinflated

    Vinflated = s^3 Vo
    Vdeflated = (2 - s)^3 Vo

    Therefore the scaling factor s can be estimated from the equation:

    s^3 (2 - a) - 6 s^2 + 12 s - 8 = 0

    Which has three roots, one real and two complex.
    """
    p = [2.0 - overlap_factor, -6.0, 12.0, -8.0]

    r = np.roots(p)

    scaling_factor = np.real(r[~np.iscomplex(r)])[0]

    predicted = (scaling_factor**3 - (2.0 - scaling_factor) ** 3) / scaling_factor**3

    L.debug(
        "Overlap Factor: %.3f, Scaling Factor: %.3f, Predicted Overlap: %.3f",
        overlap_factor,
        scaling_factor,
        predicted,
    )

    return scaling_factor * (points - centroid) + centroid


def convex_polygon_with_overlap2(centroid, points, overlap_factor):
    """Given the centroid of a convex polygon and its points,
    uniformly dilate in order to expand by the overlap factor
    overlap_factor = (Vnew - Vold) / Vold

    It can be proven that uniform scaling with a factor s
    of a polygon sitting on the origin cubicly increases its
    volume:

    Vnew = s^3 Vold

    Therefore we can determine how much to scale every point
    in order to achieve the desired overlap by using the two
    relationships abovel.

    overlap_factor = (s^3 V_old - V_old) / V_old = s^3 - 1

    Returns the new point array with the scaled points.
    """
    scaling_factor = (overlap_factor + 1.0) ** (1.0 / 3.0)
    return scaling_factor * (points - centroid) + centroid
