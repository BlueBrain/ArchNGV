""" Connect astrocytes with vasculature targets
"""

import logging

import numpy as np
import scipy.stats
import scipy.sparse

from spatial_index import point_rtree
from archngv.spatial import collision


L = logging.getLogger(__name__)


def _num_endfeet_distribution(mean, std, clip_a, clip_b):
    """ Returns the truncated normal distribution with (mean, std)
    minimum value clip_a and maximum value clip_b
    """
    a, b = (clip_a - mean) / std, (clip_b - mean) / std

    return scipy.stats.truncnorm(a, b, mean, std)


def _filter_according_to_strategy(domain_position,
                                  graph_positions,
                                  number_of_endfeet,
                                  reachout_strategy_function):

    n_targets = len(graph_positions)

    if number_of_endfeet < n_targets:

        distances = np.linalg.norm(domain_position - graph_positions, axis=1)

        if number_of_endfeet == 1:

            idx = np.array([[np.argmin(distances)]])

        else:

            idx = reachout_strategy_function(distances, number_of_endfeet)

    else:

        idx = np.arange(n_targets, dtype=np.intp)

    return idx


def domains_to_vasculature(cell_ids,
                           reachout_strategy_function,
                           graph_positions,
                           graph_radii,
                           microdomains,
                           properties):
    """
    1. Generate structural connectivity from the geometrical aspects
    of hulls and target spheres.

    2. For each domain determine the number of endfeet according to a biological
    normal distribution

    3. Using a reachout strategy select twhich ones to keep

    """
    L.info('Endfeet Distribution Paremeters %s', properties['endfeet_distribution'])

    n_distr = _num_endfeet_distribution(*properties['endfeet_distribution'])

    domain_target_edges = []
    index = point_rtree(graph_positions)

    for domain_index, cell_id in enumerate(cell_ids):

        number_of_endfeet = np.round(n_distr.rvs()).astype(np.int)

        if number_of_endfeet == 0:
            continue

        domain = microdomains[int(cell_id)]
        idx = index.intersection(*domain.bounding_box)

        if idx.size == 0:
            continue

        mask = collision.convex_shape_with_spheres(domain.face_points,
                                                   domain.face_normals,
                                                   graph_positions[idx],
                                                   graph_radii[idx])
        if not mask.any():
            continue

        idx = idx[mask]

        sliced = _filter_according_to_strategy(domain.centroid,
                                               graph_positions[idx],
                                               number_of_endfeet,
                                               reachout_strategy_function)

        idx = idx[sliced]
        domain_target_edges.extend([(domain_index, target_index) for target_index in idx])

    return np.asarray(domain_target_edges)
