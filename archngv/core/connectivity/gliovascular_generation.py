""" Facade classes for NGV connectivity
"""

# pylint: disable = no-name-in-module

import logging
import numpy as np

from archngv.core.connectivity.detail.gliovascular_generation.graph_reachout import strategy
from archngv.core.connectivity.detail.gliovascular_generation.graph_targeting import create_targets
from archngv.core.connectivity.detail.gliovascular_generation.\
    graph_connect import domains_to_vasculature
from archngv.core.connectivity.detail.gliovascular_generation.\
    surface_intersection import surface_intersect


L = logging.getLogger(__name__)


def generate_gliovascular(cell_ids,
                          astrocytic_positions,
                          astrocytic_domains,
                          vasculature,
                          params):
    """ For each astrocyte id find the connections to the vasculature

    Args:
        cell_ids: array[int, (N,)]
        astrocyte_positions: array[float, (N, 3)]
        astrocytic_domains: MicrodomainTesselation
        vasculature: Vasculature
        params: gliovascular parameters dict

    Returns:
        endfeet_positions: array[float, (M, 3)]
        graph_positions: array[float, (M, 3)]
        endfeet_to_astrocyte_mapping: array[int, (M,)]
        endfeet_to_vasculature_mapping: array[int, (M,)]
    """
    L.info('STEP 1: Generation of potential targets started.')

    graph_positions, graph_vasculature_segment = create_targets(
        vasculature.points,
        vasculature.edges,
        params['graph_targeting']
    )

    L.info('STEP 1: Generation of potential targets completed.')
    L.info('%s potential targets generated.', len(graph_positions))
    L.debug('Parameters: %s', params['graph_targeting'])
    L.debug('Positions: %d\nVasculature Edges: %d', graph_positions,
                                                    graph_vasculature_segment)

    L.info('STEP 2: Connection of astrocytes with vasculature started.')

    astrocyte_graph_edges = domains_to_vasculature(
        cell_ids,
        strategy(params['connection']['reachout_strategy']),
        graph_positions,
        astrocytic_domains,
        params['connection']
    )

    L.info('STEP 2: Connection of astrocytes with vasculature completed.')
    L.info('Astro to Vasculature Connections: %d', len(astrocyte_graph_edges))

    L.info('STEP 3: Mapping from graph points to vasculature surface started.')

    segments_beg, segments_end = vasculature.segments
    sg_radii_beg, sg_radii_end = vasculature.segments_radii

    astrocyte_idx, graph_target_idx = astrocyte_graph_edges.T

    (
        endfeet_positions,
        endfeet_to_astrocyte_mapping,
        endfeet_to_vasculature_mapping
    ) = surface_intersect(
        astrocytic_positions.astype(np.float64),
        graph_positions.astype(np.float64),
        segments_beg.astype(np.float64),
        segments_end.astype(np.float64),
        sg_radii_beg.astype(np.float64),
        sg_radii_end.astype(np.float64),
        astrocyte_idx.astype(np.uintp),
        graph_target_idx.astype(np.uintp),
        graph_vasculature_segment.astype(np.uintp),
        vasculature.edges.astype(np.uintp),
        vasculature.point_graph
    )

    L.info('STEP 3: Mapping from graph points to vasculature surface completed.')
    L.debug('Results:\n Endfeet Positions: %d\ne2a: %d\ne2v: %d', endfeet_positions,
                                                                  endfeet_to_astrocyte_mapping,
                                                                  endfeet_to_vasculature_mapping)
    return (endfeet_positions,
            graph_positions,
            endfeet_to_astrocyte_mapping,
            endfeet_to_vasculature_mapping)
