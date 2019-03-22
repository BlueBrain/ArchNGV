
from ...data_structures.data_cells import CellData
from ...data_structures.data_gliovascular import GliovascularData
from ...data_structures.data_synaptic import SynapticData
from ...data_structures.data_microdomains import MicrodomainTesselation
from ...data_structures.connectivity_gliovascular import GliovascularConnectivity
from ...data_structures.connectivity_neuroglial import NeuroglialConnectivity

import logging


L = logging.getLogger(__name__)


def obtain_endfeet_data(ngv_config, astrocyte_index):
    """ Extract the endfeet information from astrocyte_index if any, otherwise return None
    """
    gv_data_path = ngv_config.output_paths('gliovascular_data')
    gv_conn_path = ngv_config.output_paths('gliovascular_connectivity')

    with GliovascularData(gv_data_path) as gv_data, \
         GliovascularConnectivity(gv_conn_path) as gv_conn:

        endfeet_indices = gv_conn.astrocyte.to_endfoot(astrocyte_index)

        if len(endfeet_indices) == 0:
            L.warning('No endfeet found for astrocyte index {}'.format(astrocyte_index))
            return None
        else:
            L.debug('Found endfeet {} for astrocyte index {}'.format(endfeet_indices, astrocyte_index))

        targets = gv_data.endfoot_surface_coordinates[sorted(endfeet_indices)]

    L.debug('Endfeet Targets: {}'.format(targets))
    return targets


def obtain_cell_properties(ngv_config, astrocyte_index):
    """ Extract the cell info (cell_name, pos and radius) and its microdomain
    via the ngv_config and its index.
    """
    cell_data_path = ngv_config.output_paths('cell_data')

    with CellData(cell_data_path) as cell_data:

        cell_name = str(cell_data.astrocyte_names[astrocyte_index], 'utf-8')
        soma_position = cell_data.astrocyte_positions[astrocyte_index]
        soma_radius = cell_data.astrocyte_radii[astrocyte_index]

    with MicrodomainTesselation(ngv_config.output_paths('overlapping_microdomain_structure')) as md_fd:
        microdomain = md_fd.domain_object(astrocyte_index)

    L.debug('Index: {}, Name: {}, Pos: {}, Rad: {}'.format(astrocyte_index, cell_name, soma_position, soma_radius))
    return cell_name, soma_position, soma_radius, microdomain


def obtain_synapse_data(ngv_config, astrocyte_index):
    """ Obtain the synapse coordinates that correspond to the microdomain
    of astrocyte_index
    """
    sn_data_path = ngv_config.output_paths('synaptic_data')
    ng_conn_path = ngv_config.output_paths('neuroglial_connectivity')

    with SynapticData(sn_data_path) as sn_data, \
         NeuroglialConnectivity(ng_conn_path) as ng_conn:

        sorted_synapse_idx = sorted(ng_conn.astrocyte.to_synapse(astrocyte_index))
        synapse_positions = sn_data.synapse_coordinates[sorted_synapse_idx]

    L.debug('Number of synapses for astro index {}: {}'.format(astrocyte_index, len(synapse_positions)))
    return synapse_positions

