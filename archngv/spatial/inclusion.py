# SPDX-License-Identifier: Apache-2.0

""" Geometric inclusion checks
"""
import numpy as np


def spheres_in_sphere(centers, radii, center, radius):
    """Checks if spheres are included in sphere"""
    values = np.linalg.norm(center - centers, axis=1) - (radius - radii)
    return (values < 0.0) | np.isclose(values, 0.0)


def points_in_rectangle(points, min_point, max_point):
    """
    Args:
        points: array[float, (N, M)]
        min_point: array[float, (M,)]
        max_point: array[float, (M, )]

    Returns: array[bool, (N,)]
    """
    greater_or_equal = (min_point < points) | np.isclose(min_point, points)
    smaller_or_equal = (points < max_point) | np.isclose(max_point, points)
    return np.all(np.logical_and(smaller_or_equal, greater_or_equal), axis=1)


def spheres_in_rectangle(sphere_centers, sphere_radii, min_point, max_point):
    """
    Args:
        sphere_centers: array[float, (N, M)]
        sphere_radii: array[float, (N,)]
        min_point: array[float, (M,)]
        max_point: array[float, (M, )]

    Returns: array[bool, (N,)]
    """

    radii_expanded = sphere_radii[:, np.newaxis]

    diff = sphere_centers - radii_expanded
    summ = sphere_centers + radii_expanded

    greater_or_equal = (min_point < diff) | np.isclose(min_point, diff)
    smaller_or_equal = (summ < max_point) | np.isclose(max_point, summ)
    return np.all(np.logical_and(greater_or_equal, smaller_or_equal), axis=1)
