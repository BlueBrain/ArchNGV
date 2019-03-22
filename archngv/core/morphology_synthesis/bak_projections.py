import os
import logging
import numpy as np

from .detail.random_walk import biased_random_walk
from .detail.morphology_types import MAP_TO_NEURONAL

from ..data_structures.data_cells import CellData
from ..data_structures.data_gliovascular import GliovascularData
from ..data_structures.connectivity_gliovascular import GliovascularConnectivity

from morphio import SectionType, PointLevel
from morphio.mut import Morphology

L = logging.getLogger(__name__)


def _point_on_surface(center, radius, target):
    return center + (target - center) * radius / np.linalg.norm(center - target)


def grow_endfeet_projections(soma_position, soma_radius, endfeet_targets, parameters):

    randomness = np.clip(parameters['randomness'], 0.0, 1.0)
    segment_length = parameters['segment_length']

    for endfoot_target in endfeet_targets:

        start_point_surface = \
        _point_on_surface(soma_position, soma_radius, endfoot_target)

        start_point_data = np.hstack((start_point_surface, 1.))

        point_data = \
        biased_random_walk(start_point_data, endfoot_target, randomness, segment_length)

        yield point_data


def _generate_morphology(filepath, soma_position, soma_radius, points_per_process):

    astrocyte = Morphology()
    astrocyte.soma.points = np.array([soma_position.tolist()])
    astrocyte.soma.diameters = np.array([[soma_radius]])

    endfoot_type = SectionType(2)

    for morphology_point_data in points_per_process:

        point_list = morphology_point_data[:, :3].tolist()
        diamt_list = (2.0 * morphology_point_data[:, 3]).tolist()

        astrocyte.append_root_section(PointLevel(point_list, diamt_list), endfoot_type)

    # write to file
    astrocyte.write(filepath)


def _get_astrocyte_data(ngv_config, astrocyte_index):

    with GliovascularData(ngv_config.output_paths('gliovascular_data')) as gv_data, \
         GliovascularConnectivity(ngv_config.output_paths('gliovascular_connectivity')) as gv_conn:
        endfeet_indices = gv_conn.astrocyte.to_endfoot(astrocyte_index)

        if len(endfeet_indices) == 0:
            L.warning('No endfeet found for astrocyte index {}'.format(astrocyte_index))
            return None
        else:
            L.debug('Found endfeet {} for astrocyte index {}'.format(endfeet_indices, astrocyte_index))

        targets = gv_data.endfoot_surface_coordinates[sorted(endfeet_indices)]

    with CellData(ngv_config.output_paths('cell_data')) as cell_data:

        cell_name = str(cell_data.astrocyte_names[astrocyte_index], 'utf-8')
        soma_position = cell_data.astrocyte_positions[astrocyte_index]
        soma_radius = cell_data.astrocyte_radii[astrocyte_index]

    parameters = ngv_config.parameters['synthesis']

    return targets, soma_position, soma_radius, cell_name, parameters


def _synthesize_endfeet_for_astrocyte(args):

    astrocyte_index, ngv_config = args

    data = _get_astrocyte_data(ngv_config, astrocyte_index)

    if data is not None:

        endfeet_targets, soma_position, soma_radius, cell_name, parameters = data

        process_gen = \
        grow_endfeet_projections(soma_position, soma_radius, endfeet_targets, parameters)

        output_file = os.path.join(ngv_config.morphology_directory, '{}.h5'.format(cell_name))

        _generate_morphology(output_file, soma_position, soma_radius, process_gen)


def synthesize_astrocyte_endfeet(ngv_config, astrocyte_ids, apply_func):

    def data_generator(ngv_config, ids):
        for astro_id in ids:
            yield astro_id, ngv_config

    apply_func(_synthesize_endfeet_for_astrocyte, data_generator(ngv_config, astrocyte_ids))

