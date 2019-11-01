""" Facade classes for NGV connectivity
"""

# pylint: disable = no-name-in-module

import logging
import pandas as pd
import numpy as np

from archngv.building.connectivity.detail.gliovascular_generation.graph_reachout import strategy
from archngv.building.connectivity.detail.gliovascular_generation.graph_targeting import create_targets
from archngv.building.connectivity.detail.gliovascular_generation.\
    graph_connect import domains_to_vasculature
from archngv.building.connectivity.detail.gliovascular_generation.\
    surface_intersection import surface_intersect


L = logging.getLogger(__name__)


def _create_point_sampling_on_vasculature_skeleton(vasculature, graph_targeting_params):

    graph_positions, graph_vasculature_segment = create_targets(
        vasculature.points,
        vasculature.edges,
        graph_targeting_params
    )

    graph_vasculature_sections = vasculature.map_edges_to_sections[graph_vasculature_segment]

    L.info('STEP 1: Generation of potential targets completed.')
    L.info('%s potential targets generated.', len(graph_positions))
    L.debug('Parameters: %s', graph_targeting_params)
    L.debug('Positions: %d\nVasculature Edges: %d', graph_positions,
                                                    graph_vasculature_segment)

    L.info('STEP 2: Connection of astrocytes with vasculature started.')
    sg_radii_beg, sg_radii_end = vasculature.segments_radii

    graph_radii = np.min((sg_radii_beg[graph_vasculature_segment],
                          sg_radii_end[graph_vasculature_segment]), axis=0)

    return pd.DataFrame({
        'x': graph_positions[:, 0],
        'y': graph_positions[:, 1],
        'z': graph_positions[:, 2],
        'r': graph_radii,
        'vasculature_segment_id': graph_vasculature_segment,
        'vasculature_section_id': graph_vasculature_sections})


def _find_surface_intersections(astrocytic_positions, skeleton_seeds, astrocyte_graph_edges, vasculature):

    segments_beg, segments_end = vasculature.segments
    astrocyte_idx, graph_target_idx = astrocyte_graph_edges.T

    sg_radii_beg, sg_radii_end = vasculature.segments_radii

    return surface_intersect(
        astrocytic_positions.astype(np.float64),
        skeleton_seeds.loc[:, ('x', 'y', 'z')].values.astype(np.float64),
        segments_beg.astype(np.float64),
        segments_end.astype(np.float64),
        sg_radii_beg.astype(np.float64),
        sg_radii_end.astype(np.float64),
        astrocyte_idx.astype(np.uint64),
        graph_target_idx.astype(np.uint64),
        skeleton_seeds['vasculature_segment_id'].values.astype(np.uint64),
        vasculature.edges.astype(np.uint64),
        vasculature.point_graph
    )


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

    skeleton_seeds = _create_point_sampling_on_vasculature_skeleton(vasculature, params['graph_targeting'])

    astrocyte_skeleton_connectivity = domains_to_vasculature(
        cell_ids,
        strategy(params['connection']['reachout_strategy']),
        skeleton_seeds,
        astrocytic_domains,
        params['connection']
    )

    L.info('STEP 2: Connection of astrocytes with vasculature completed.')
    L.info('Astro to Vasculature Connections: %d', len(astrocyte_skeleton_connectivity))
    L.info('STEP 3: Mapping from graph points to vasculature surface started.')

    (
        endfeet_positions,
        endfeet_to_astrocyte_mapping,
        endfeet_to_vasculature_mapping
    ) = _find_surface_intersections(
        astrocytic_positions, skeleton_seeds,
        astrocyte_skeleton_connectivity,
        vasculature)

    L.info('STEP 3: Mapping from graph points to vasculature surface completed.')
    L.debug('Results:\n Endfeet Positions: %d\ne2a: %d\ne2v: %d', endfeet_positions,
                                                                  endfeet_to_astrocyte_mapping,
                                                                  endfeet_to_vasculature_mapping)
    return (endfeet_positions,
            skeleton_seeds[['x', 'y', 'z']].values,
            endfeet_to_astrocyte_mapping,
            endfeet_to_vasculature_mapping)
