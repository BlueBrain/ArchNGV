""" Geometric Collisions Module
"""

# pylint: disable = too-many-arguments

import numpy as np

from archngv.utils.linear_algebra import rowwise_dot

from .gjk_algorithm import GJK


def convex_shape_with_convex_shape(shape1, shape2):
    """Returns true if shape1 and shap2 collide"""
    return GJK(shape1, shape2, 10)


def sphere_with_spheres(center, radius, centers, radii):
    """
    Returns:
        1D array: Bool
            True for i-th row if sphere (center, radius) collides with (centers[i], radii[i])
    """
    values = np.linalg.norm(center - centers, axis=1) - (radii + radius)
    return (values < 0.0) | np.isclose(values, 0.0)


def sphere_with_sphere(sphere1_data, sphere2_data):
    """
    Args:
        sphere1_data, sphere2_data: 1D array
            True if (x1, y1, z1, r1) sphere collides with (x2, y2, z2, r2) sphere.
    """
    x_1, y_1, z_1, r_1 = sphere1_data
    x_2, y_2, z_2, r_2 = sphere2_data

    return np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2 + (z_2 - z_1) ** 2) <= r_1 + r_2


def sphere_with_capsule(center, radius, p_0, p_1, r_0, r_1):
    """Returns true if sphere (center, radius) collides with capsule (p0, r0, p1, r1)"""
    vec = p_1 - p_0

    s_2 = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2

    # t for closest distance between line and point
    tau = np.dot(center - p_0, vec) / s_2

    # clamp for segment extent
    tau = np.clip(tau, 0.0, 1.0)

    # closest point on capsule axis to point
    p_t = p_0 + tau * vec

    # distance of closest point to sphere center
    closest_distx = np.linalg.norm(p_t - center)

    # radius to varying radius capsule for projection to point p_t
    r_c = r_0 + (r_1 - r_0) * tau

    return (closest_distx < radius + r_c) & ~np.isclose(closest_distx, radius + r_c)


def sphere_with_capsules(center, radius, p0s, p1s, r0s, r1s):
    """
    Returns:
        1D array: Bool
            i-th pos true if sphere (center, r) collides with (p0s[i], r0s[i], p1s[i], r1s[i])
    """
    vectors = p1s - p0s

    s_2 = vectors[:, 0] ** 2 + vectors[:, 1] ** 2 + vectors[:, 2] ** 2

    # t for closest distance between line and point
    tau = rowwise_dot(center - p0s, vectors) / s_2

    # clamp for segment extent
    tau = np.clip(tau, 0.0, 1.0)

    # closest point on capsule axis to point
    p_t = p0s + tau[:, np.newaxis] * vectors

    # distance of closest point to sphere center
    closest_distx = np.linalg.norm(p_t - center, axis=1)

    # radius to varying radius capsule for projection to point p_t
    # mask = ~np.isclose(r0s, r1s)

    r_c = r0s + (r1s - r0s) * tau

    return (closest_distx < radius + r_c) & ~np.isclose(closest_distx, radius + r_c)


def convex_shape_with_point(face_points, face_normals, point):
    """Assumes that normals point outward"""
    sgn_distx = rowwise_dot(point - face_points, face_normals)
    return np.all((sgn_distx < 0.0) | np.isclose(sgn_distx, 0.0))


def convex_shape_with_spheres(face_points, face_normals, target_positions, target_radii=None):
    """
    Example of one side of the convex polygon:
    AP.n positive -> outside of side AB
    A (face_point: one per face)
    |\
    | \
    |  \
    |   P (sphere center to check: target_positions)
    |
    |--> n (face_normal)
    |
    |
    |
    |
    B (face_point: not needed if A is available for the check)

    """
    # expand a dimension to do all the pairwise differences
    # resulting dimensions for k_m (N, M, 3)
    k_m = target_positions - face_points[:, np.newaxis]

    # signed distance from each point to each side of the domain (N, M)
    signed_dist = np.einsum("ijk, ik -> ji", k_m, face_normals)

    # if there is at least one signed distance from a point to a triangle side
    # that remains positive after subtracting the radius then that point belongs
    # to the halfspace outside that side of the triangle, thus not possible to intersect
    if target_radii is not None:
        signed_dist -= target_radii[:, np.newaxis]

    # fix accuracy issues
    signed_dist[np.isclose(signed_dist, 0.0)] = 0.0

    return ~np.any(signed_dist > 0.0, axis=1)
