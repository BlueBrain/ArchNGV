""" Facade classes for NGV connectivity
"""
import logging
import collections
from itertools import groupby

import numpy
import numpy as np

from .detail.gliovascular_generation import check
from .detail.gliovascular_generation.graph_reachout import strategy
from .detail.gliovascular_generation.graph_targeting import create_targets as _create_potential_targets_on_vasculature
from .detail.gliovascular_generation.graph_connect import domains_to_vasculature as _connect_astrocytes_with_vasculature_graph
from .detail.gliovascular_generation.surface_intersection import surface_intersect


L = logging.getLogger(__name__)


def _validate_input(astrocytic_positions, astrocytic_domains, vasculature, options):

    L.info('Validating input started.')

    check.equal_length(astrocytic_positions, astrocytic_domains)
    check.keys(('graph_targeting', 'connection'), options)
    check.points_inside_polyhedra(astrocytic_positions, astrocytic_domains)

    L.info('Validating input completed.')


def generate_gliovascular(cell_ids,
                          astrocytic_positions,
                          vasculature,
                          ngv_config, map_func):

    options = ngv_config.parameters['gliovascular_connectivity']

    microdomains_filepath = ngv_config.output_paths('overlapping_microdomain_structure')

    L.info('STEP 1: Generation of potential targets started.')

    graph_positions,\
    graph_vasculature_segment = \
    _create_potential_targets_on_vasculature(

        vasculature.points,
        vasculature.edges,
        options['graph_targeting']

    )

    L.info('STEP 1: Generation of potential targets completed.')
    L.info('{} potential targets generated.'.format(len(graph_positions)))
    L.debug('Parameters: {}'.format(options['graph_targeting']))
    L.debug('Positions: {}\nVasculature Edges:{}'.format(graph_positions,
                                                         graph_vasculature_segment))

    L.info('STEP 2: Connection of astrocytes with vasculature started.')

    astrocyte_graph_edges = \
    _connect_astrocytes_with_vasculature_graph(

        cell_ids,
        vasculature,
        strategy(options['connection']['Reachout Strategy']),
        graph_positions,
        graph_vasculature_segment,
        microdomains_filepath,
        options['connection']

    )

    L.info('STEP 2: Connection of astrocytes with vasculature completed.')
    L.info('Astro to Vasculature Connections: {}'.format(len(astrocyte_graph_edges)))

    L.info('STEP 3: Mapping from graph points to vasculature surface started.')

    segments_beg, segments_end = vasculature.segments
    sg_radii_beg, sg_radii_end = vasculature.segments_radii

    astrocyte_idx, graph_target_idx = astrocyte_graph_edges.T

    endfeet_positions,\
    endfeet_to_astrocyte_mapping,\
    endfeet_to_vasculature_mapping = \
    surface_intersect(
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
    L.debug('Results:\n Endfeet Positions: {}\ne2a: {}\ne2v: {}'.format(endfeet_positions,
                                                                        endfeet_to_astrocyte_mapping,
                                                                        endfeet_to_vasculature_mapping))

    return (endfeet_positions,
            graph_positions,
            endfeet_to_astrocyte_mapping,
            endfeet_to_vasculature_mapping)

