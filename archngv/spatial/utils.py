""" Morphspatial utilities
"""

from itertools import chain

import numpy as np

from archngv.math_utils.linear_algebra import rowwise_dot
from archngv.math_utils.geometry import rotate_from_unit_vector_to_another


def _globally_ordered_verts(face_points, face_vertices):
    """ Given a chain of face_vertices, find how to traverse it
    via face_point coordinate ordering in order to achieve the same
    ordering from any overlapping polygon
    """
    # the start vertex of the face is selected be the
    # ordering of the face coordinates by zyx
    first_index, second_index = np.lexsort(face_points.T)[:2]

    n_verts = len(face_points)

    if first_index > second_index or \
       (first_index == 0 and second_index == n_verts - 1):
        # if the second index is smaller than the first it means that we need to
        # iterate the vertices backwards starting from the second_index
        # By shifting the array by len(face_vertices) - second_index - 1 we make
        # sure that the second_index is at the end of the array eg
        # [4, 5, 6, 7, 8] -> [7, 8, 4, 5, 6] if first_index = 2, second_index = 1
        # Then we reverse the array to [6, 5, 4, 8, 7] to achieve the same ordering
        right_shift = len(face_vertices) - first_index - 1
        return np.roll(face_vertices, right_shift)[::-1]

    # shoft the array to the left so that first_index element is first eg
    # [5, 6, 7, 8] -> [7, 8, 5, 6] if first_index = 2
    return np.roll(face_vertices, -first_index)


def triangles_from_polygons_generator(points, face_vertices_collection):
    """ Triangles from polygons
    """
    for vlist in face_vertices_collection:

        n_vertices = len(vlist)

        if n_vertices == 3:

            yield vlist

        else:

            face_vertices = np.asarray(vlist)

            face_points = points[face_vertices]
            o_verts = _globally_ordered_verts(face_points, face_vertices)

            for index in range(2, n_vertices):
                yield [o_verts[0], o_verts[index - 1], o_verts[index]]


def fromiter2D(gen, number_of_columns, dtype):  # pylint: disable = invalid-name
    """ Generate 2D array from generator
    """
    raveled_data = np.fromiter(chain.from_iterable(gen), dtype=dtype)
    return raveled_data.reshape((len(raveled_data) // number_of_columns, number_of_columns))


def are_normals_backward(centroid, points, triangles, normals):
    """ Check which normals point towards the inside of the convex hull
    """
    vectors = points[triangles[:, 0]] - centroid

    signed_distx = rowwise_dot(vectors, normals)

    return (signed_distx < 0.) & ~np.isclose(signed_distx, 0.)


def make_normals_outward(centroid, points, triangles, normals):
    """ Normals that point inwards are flipped
    """
    new_triangles = triangles.copy()

    backward = are_normals_backward(centroid, points, triangles, normals)

    new_triangles[backward] = np.fliplr(triangles[backward])

    return new_triangles


def are_in_the_same_side(vectors1, vectors2):
    """ Check if vectors point to the same halfspace
    """
    return rowwise_dot(vectors1, vectors2) > 0.


# pylint: disable = too-many-locals
def create_contact_sphere_around_truncated_cylinder(p_0, p_1, r_0, r_1, n_spheres=1):
    """ Create a spheres that touches a truncated cylinder
    """
    taus = np.random.random(size=n_spheres)

    phi = np.random.uniform(0., 2. * np.pi, size=n_spheres)

    vec = p_1 - p_0

    rot_m = rotate_from_unit_vector_to_another(np.array([0., 0., 1.]), vec / np.linalg.norm(vec))

    p_t = p_0 + vec * taus[:, np.newaxis]

    r_t = r_0 + (r_1 - r_0) * taus

    radii = np.sqrt(np.random.random(size=n_spheres)) * (2. - 1.) + 1.
    r_s = r_t + radii

    length = np.linalg.norm(p_t - p_0, axis=1)

    cs_phi = np.cos(phi)
    sn_phi = np.sin(phi)

    coo_xs = r_s * (cs_phi * rot_m[0, 0] + sn_phi * rot_m[0, 1]) + length * rot_m[0, 2]
    coo_ys = r_s * (cs_phi * rot_m[1, 0] + sn_phi * rot_m[1, 1]) + length * rot_m[1, 2]
    coo_zs = r_s * (cs_phi * rot_m[2, 0] + sn_phi * rot_m[2, 1]) + length * rot_m[2, 2]

    return (p_0[0] + coo_xs, p_0[1] + coo_ys, p_0[2] + coo_zs, radii)
