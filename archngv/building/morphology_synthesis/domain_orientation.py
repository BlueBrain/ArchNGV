""" Functions related to the calculation of the trunk orientation
from the geometry of the microdomain
"""
import logging

import numpy as np

from archngv.utils.linear_algebra import angle_matrix, normalize_vectors
from archngv.utils.ngons import subdivide_triangles_by_total_area

L = logging.getLogger(__name__)


def orientations_from_domain(
    soma_center, domain_points, domain_triangles, n_trunks, fixed_targets=None
):
    """
    Given the triangular mesh of a convex domain, select up to n_trunks orientations.
    The triangles are subdivided in to smaller, nested ones that provide a better sampling
    of orientations.

    The orientations are selected sequentially, taking into account the orientations from
    the fixed_targets and the ones that are selected incrementally. Given a candidate orientation
    the mean angle to all the existing orientations is maximized (max -> antiparallel) and there is
    also preference towards longer vectors, in order to follow the elongation of the domain.

    Args:
        soma_center: array[float, (3,)]
            The center from which the orientations are going to be calculated. It is not
            necessarily the same as the centroid of the convex domain (it can be though).
        domain_points: array[float, (N, 3)]
            Array of the domain points.
        domain_triangles: array[int, (M, 3)]
            Array of integer triplets corresponding to the triangles of the domain, the
            coordinates of which can be accessed from domain_points.
        n_trunks: int
            Number of processes to create (without fixed_targets)
        fixed_targets: list[array[float, (3,)]]
            Points that are predetermined. For example endfeet targets have already
            defined orientations. We have to take them into account if given.

    Returns: array[float, (n_trunks, 3)], array[float, (n_trunks,)]
       The selected orientations and their respective vector norms
    """
    points, _ = subdivide_triangles_by_total_area(domain_points, domain_triangles, 10 * n_trunks)

    if fixed_targets is not None:
        points = remove_overlapping_orientations(soma_center, points, fixed_targets)

    vectors = points - soma_center
    lengths = np.linalg.norm(vectors, axis=1)
    orientations = vectors / lengths[:, np.newaxis]

    objective_function = Objective(soma_center, points, fixed_targets)
    chosen_idx = choose_ids(n_trunks, points, objective_function)

    L.debug("Number of idx calculated: %d", len(chosen_idx))
    return orientations[chosen_idx], lengths[chosen_idx]


def remove_overlapping_orientations(ref_point, points, to_remove_points):
    """Return the points, the orientations of which do not overlap with
    the orientations of to_remove_points. Usualy, the number of orientations
    is small, so we do it the simple/expensive way, by combinatorial comparison.

    Args:
        ref_point: array[float, (3,)]
            Reference point for the calculation of the orientations.
        points: array[float, (N, 3)]
            Points the orientations of which will be checked.
        to_remove_poitnts: array[float, (M, 3)]
            Points the orientations of which we want to avoid overlapping with
            the orientations of points.

    Returns:
        The cleaned points without overlapping orientations.
    """
    to_remain = np.ones(len(points), dtype=bool)

    orientations = normalize_vectors(ref_point - points)
    fixed_orientations = normalize_vectors(ref_point - to_remove_points)

    # remove overlaps with predetermined orientations
    for i, orientation in enumerate(orientations):
        for existing_orientation in fixed_orientations:
            if np.allclose(existing_orientation, orientation):
                to_remain[i] = False

    return points[to_remain]


def quadratic_angles(vectors1, vectors2):
    """Quadratic angles"""
    return (angle_matrix(vectors1, vectors2) - np.pi) ** 2 / np.pi**2


class Objective:
    """
    Args:
        center: array[float, 3]
        points: array[float, (N, 3)]
        fixed_points: array[float, (M, 3)]

    Attrs:
        angle_matrix: array[float, (N, N)]
            Pairwise angles of points - center.
        fixed_angle_matrix_sum: array[float, (N, 1)]
            Sum of angles of all  orientations
            with all the orientations from the fixed points
        n_fixed_points: int
            Number of fixed points. Zero if None.
    """

    def __init__(self, center, points, fixed_points=None):

        vectors = points - center
        lengths = np.linalg.norm(vectors, axis=1)
        self.angles = quadratic_angles(vectors, vectors)

        max_length = lengths.max()
        self.weights = (max_length - lengths) / max_length

        if fixed_points is not None:
            self.fixed_sum = quadratic_angles(vectors, fixed_points - center).sum(axis=1)
            self.n_fixed_points = len(fixed_points)
        else:
            self.fixed_sum = None
            self.n_fixed_points = 0

    def __call__(self, index, selected_ids):
        """Score of the angles from index orientation and the occupied selected ids,
        plus the angles to the fixed orientations, plus the normalized length weight.

        A 95% priority is given to the angle cost contribution, and a 5% to the vector
        length bias.

        Minimization leads to big angles with all the occupied and fixed target orientations
        and a bias to choose longer vectors towards the long ends of the domains geometry.
        """
        score_angles = 0.0

        if self.fixed_sum is not None:
            score_angles += self.fixed_sum[index]

        score_angles += np.sum(self.angles[index, selected_ids])
        total = len(selected_ids) + self.n_fixed_points

        if total > 0:
            score_angles /= total

        return 0.95 * score_angles + 0.05 * self.weights[index]


def choose_ids(total_number, points, objective):
    """Choose the domain orientations that maximize the objective function
    Args:
        total_number: int
            Total number of ids to choose. If total_number >= available_ids
            then all available_ids are returned.
        available_ids: set[int]
        objective_function: function
            Maximize function(available_id)

    Returns: array[int]
        The total_number ids form the available_ids that maximize the
        objective_function.
    """
    available_ids = set(range(len(points)))

    if total_number >= len(available_ids):
        return np.asarray(list(available_ids), dtype=np.int32)

    selected = []

    for _ in range(total_number):

        best_index = next(iter(available_ids))
        best_cost = objective(best_index, selected)
        for index in available_ids:
            cost = objective(index, selected)
            if cost < best_cost:
                best_cost = cost
                best_index = index

        selected.append(best_index)
        available_ids.remove(best_index)

    return np.asarray(list(selected), dtype=np.int32)
