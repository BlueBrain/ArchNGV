""" Functions related to the calculation of the trunk orientation
from the geometry of the microdomain
"""

import logging
import numpy as np


L = logging.getLogger(__name__)


def orientations_from_domain(soma_center,
                             domain,
                             n_trunks,
                             fixed_targets=None,
                             face_interpolation=True):
    """ Given the domain shape and the number of apical directions, find
    from the trunk directions. It uses kmeans of the vectors from the center
    to the convex hull vertices and from the resulting classes takes the furthest
    one.
    Returns the normalized directions for the apical trunks.

    Args:
        soma_center: array[float, (3,)]
            The center of the soma, not the centroid of the microdomain.
        domain: ConvexPolygon
            Microdomain object from MorphSpatial
        n_trunks: int
            Number of processes to create (including the fixed ones)
        fixed_targets: list[array[float, (3,)]]
            Orientations that are predetermined. For example endfeet targets have already
            defined orientations. We have to take them into account if given.
        face_interpolation: bool
            If enabled it adds the centroids of the faces into the point list in order to create
            more orientations to choose from.
    """
    if face_interpolation:
        domain_vectors = interpolate_triangles(domain.points, domain.triangles) - soma_center
    else:
        domain_vectors = domain.points - soma_center

    domain_vector_lengths = np.linalg.norm(domain_vectors, axis=1)
    domain_orientations = domain_vectors / domain_vector_lengths[:, np.newaxis]

    if fixed_targets is not None and len(fixed_targets) > 0:

        fixed_orientations = fixed_targets - soma_center
        fixed_orientations /= np.linalg.norm(fixed_orientations, axis=1)[:, np.newaxis]

        # smallest angle to a fixed orientation
        objective_function = lambda index: min(
            vector_angle(domain_orientations[index], fixed_orientation)
            for fixed_orientation in fixed_orientations
        )

    else:

        fixed_orientations = []
        # length of vector reflects the anisotropy of the domain
        objective_function = lambda index: domain_vector_lengths[index]

    chosen_idx = choose_orientations(n_trunks,
                                     domain_orientations,
                                     fixed_orientations,
                                     objective_function)

    L.debug('Number of idx calculated: %d', len(chosen_idx))

    return domain_orientations[chosen_idx], domain_vector_lengths[chosen_idx]


def interpolate_triangles(points, triangles):
    """ Add face centers of triangles to the available points for the orientations
        TODO: If needed this should recursively change the connectivity as well and add the center
        of the new triangles that are generated and so on...
    """
    new_points = (points[triangles[:, 0]] + points[triangles[:, 1]] + points[triangles[:, 2]]) / 3.
    return np.vstack((points, new_points))


def available_domain_orientations(orientations, fixed_orientations):
    """ Return the indices of orientations that do not overlap with
    fixed_orientations. Usualy, the number of orientations is small, so
    we do it the simple/expensive way, by combinatorial comparison.
    """

    # all the available vectors to choose from
    available_idx = set(range(len(orientations)))

    # remove overlaps with predetermined orientations
    for i, orientation in enumerate(orientations):
        for existing_orientation in fixed_orientations:
            if np.allclose(existing_orientation, orientation):
                available_idx.remove(i)

    return available_idx


def choose_orientations(total_number, domain_orientations, fixed_orientations, objective_function):
    """ Choose the domain orientations that maximize the objective function
    """

    available = available_domain_orientations(domain_orientations, fixed_orientations)

    assert len(available) >= total_number, (available, total_number)

    chosen = set()

    for _ in range(total_number):

        best_value = 0.0
        best_index = None

        for available_index in available:

            current_value = objective_function(available_index)

            if current_value > best_value:

                best_index = available_index
                best_value = current_value

        assert best_index is not None

        chosen.add(best_index)
        available.remove(best_index)

    return np.asarray(list(chosen), dtype=np.intp)


def vector_angle(v1, v2):
    """ Angle between two vectors in radians"""
    return np.arccos(np.dot(v1, v2))
