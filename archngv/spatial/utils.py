# SPDX-License-Identifier: Apache-2.0

""" Morphspatial utilities
"""

from itertools import chain

import numpy as np

from archngv.utils.geometry import rotate_from_unit_vector_to_another
from archngv.utils.linear_algebra import rowwise_dot
from archngv.utils.ngons import vectorized_consecutive_triangle_vectors, vectorized_triangle_normal


def fromiter2D(gen, number_of_columns, dtype):  # pylint: disable = invalid-name
    """Generate 2D array from generator"""
    raveled_data = np.fromiter(chain.from_iterable(gen), dtype=dtype)
    return raveled_data.reshape((len(raveled_data) // number_of_columns, number_of_columns))


def are_normals_backward(centroid, points, triangles, normals):
    """Check which normals point towards the inside of the convex hull"""
    vectors = points[triangles[:, 0]] - centroid

    signed_distx = rowwise_dot(vectors, normals)

    return (signed_distx < 0.0) & ~np.isclose(signed_distx, 0.0)


def make_normals_outward(centroid, points, triangles):
    """Normals that point inwards are flipped"""
    new_triangles = triangles.copy()
    face_vectors = vectorized_consecutive_triangle_vectors(points, triangles)

    normals = vectorized_triangle_normal(*face_vectors)
    backward = are_normals_backward(centroid, points, triangles, normals)

    new_triangles[backward] = np.fliplr(triangles[backward])

    return new_triangles


# pylint: disable = too-many-locals
def create_contact_sphere_around_truncated_cylinder(p_0, p_1, r_0, r_1, n_spheres=1):
    """Create a spheres that touches a truncated cylinder"""
    taus = np.random.random(size=n_spheres)

    phi = np.random.uniform(0.0, 2.0 * np.pi, size=n_spheres)

    vec = p_1 - p_0

    rot_m = rotate_from_unit_vector_to_another(np.array([0.0, 0.0, 1.0]), vec / np.linalg.norm(vec))

    p_t = p_0 + vec * taus[:, np.newaxis]

    r_t = r_0 + (r_1 - r_0) * taus

    radii = np.sqrt(np.random.random(size=n_spheres)) * (2.0 - 1.0) + 1.0
    r_s = r_t + radii

    length = np.linalg.norm(p_t - p_0, axis=1)

    cs_phi = np.cos(phi)
    sn_phi = np.sin(phi)

    coo_xs = r_s * (cs_phi * rot_m[0, 0] + sn_phi * rot_m[0, 1]) + length * rot_m[0, 2]
    coo_ys = r_s * (cs_phi * rot_m[1, 0] + sn_phi * rot_m[1, 1]) + length * rot_m[1, 2]
    coo_zs = r_s * (cs_phi * rot_m[2, 0] + sn_phi * rot_m[2, 1]) + length * rot_m[2, 2]

    return (p_0[0] + coo_xs, p_0[1] + coo_ys, p_0[2] + coo_zs, radii)
