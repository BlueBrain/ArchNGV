""" Connect astrocytes with vasculature targets
"""
import logging

import numpy as np

from spatial_index import point_rtree
from archngv.spatial import collision

from archngv.utils.statistics import truncated_normal

L = logging.getLogger(__name__)


def _points_inside_domain(domain, bbox_potential_targets):
    """ Find which targets are inside the convex domain """
    points = bbox_potential_targets.loc[:, ('x', 'y', 'z')].to_numpy()
    radii = bbox_potential_targets.loc[:, 'r'].to_numpy()
    return collision.convex_shape_with_spheres(domain.face_points, domain.face_normals, points, radii)


def domains_to_vasculature(cell_ids, reachout_strategy_function, potential_targets, domains, properties):
    """
    Args:
        cell_ids: array[int, (N,)]
        reachout_strategy_function: function
        potential_targets: pandas.DataFrame
        domains: list[Microdomain]
        properties: dict

    1. Generate structural connectivity from the geometrical aspects
    of hulls and target spheres.

    2. For each domain determine the number of endfeet according to a biological
    normal distribution

    3. Using a reachout strategy select twhich ones to keep

    """
    L.info('Endfeet Distribution Paremeters %s', properties['endfeet_distribution'])

    domain_target_edges = []
    index = point_rtree(potential_targets.loc[:, ('x', 'y', 'z')].to_numpy())

    n_distr = truncated_normal(*properties['endfeet_distribution'])
    endfeet_per_domain = n_distr.rvs(size=len(cell_ids)).round().astype(np.int)

    for domain_index, cell_id in enumerate(cell_ids):

        n_endfeet = endfeet_per_domain[domain_index]

        if n_endfeet == 0:
            continue

        domain = domains[int(cell_id)]
        idx = index.intersection(*domain.bounding_box)

        if idx.size == 0:
            continue

        bbox_potential_targets = potential_targets.iloc[idx]

        are_inside_domain_mask = _points_inside_domain(domain, bbox_potential_targets)

        if not are_inside_domain_mask.any():
            continue

        domain_potential_targets = bbox_potential_targets[are_inside_domain_mask]

        if n_endfeet >= len(domain_potential_targets):
            selected = domain_potential_targets.index
        else:
            selected = reachout_strategy_function(domain.centroid, domain_potential_targets, n_endfeet)

        domain_target_edges.extend((domain_index, target_index) for target_index in selected)

    return np.asarray(domain_target_edges)