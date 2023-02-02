""" various checks for connectivity
"""

import logging

L = logging.getLogger(__name__)


def equal_length(iterable1, iterable2):
    """Check if two iterables have the same length"""
    try:
        l1 = len(iterable1)
        l2 = len(iterable2)
        assert l1 == l2

    except AssertionError as e:
        msg = "Iterables do not have same length {}, {}".format(l1, l2)
        L.error(msg)
        raise AssertionError(msg) from e


def keys(keys_to_check, dictionary):
    """Checks if list of keys are available in dictionary"""
    try:
        for key in keys_to_check:
            assert key in dictionary

    except AssertionError as e:
        msg = "{} key could not be found in config".format(key)
        L.error(msg)
        raise AssertionError(msg) from e


def points_inside_polyhedra(points, polyhedra):
    """Checks if points inside list of polyhedra.
    Args:
        points: array[float, (N, 3)]
        polyhedra: list[archngv.spatial.shapes.ConvexPolyhedron]
    """
    from archngv.spatial.collision import convex_shape_with_point

    try:
        for point, polyhedron in zip(points, polyhedra):
            assert convex_shape_with_point(polyhedron.face_points, polyhedron.face_normals, point)

    except AssertionError as e:
        msg = "Points are not inside polyhedra."
        L.error(msg)
        raise AssertionError(msg) from e
