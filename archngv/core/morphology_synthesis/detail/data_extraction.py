
from ...data_structures.data_cells import CellData
from ...data_structures.data_gliovascular import GliovascularData
from ...data_structures.data_synaptic import SynapticData
from ...data_structures.data_microdomains import MicrodomainTesselation
from ...data_structures.connectivity_gliovascular import GliovascularConnectivity
from ...data_structures.connectivity_neuroglial import NeuroglialConnectivity

import logging


L = logging.getLogger(__name__)


def obtain_endfeet_data(astrocyte_index,
                        gliovascular_data_path,
                        gliovascular_connectivity_path):
    """ Extract the endfeet information from astrocyte_index if any, otherwise return None
    """
    with \
        GliovascularData(gliovascular_data_path) as gliovascular_data, \
        GliovascularConnectivity(gliovascular_connectivity_path) as gliovascular_connectivity:

        endfeet_indices = gliovascular_connectivity.astrocyte.to_endfoot(astrocyte_index)

        if len(endfeet_indices) == 0:
            L.warning('No endfeet found for astrocyte index {}'.format(astrocyte_index))
            return None
        else:
            L.debug('Found endfeet {} for astrocyte index {}'.format(endfeet_indices, astrocyte_index))

        targets = gliovascular_data.endfoot_surface_coordinates[sorted(endfeet_indices)]

    L.debug('Endfeet Targets: {}'.format(targets))
    return targets


def obtain_cell_properties(astrocyte_index,
                           cell_data_filepath,
                           microdomains_filepath):
    """ Extract the cell info (cell_name, pos and radius) and its microdomain
    via the ngv_config and its index.
    """

    with \
        CellData(cell_data_filepath) as cell_data, \
        MicrodomainTesselation(microdomains_filepath) as microdomains:

        cell_name = str(cell_data.astrocyte_names[astrocyte_index], 'utf-8')
        soma_position = cell_data.astrocyte_positions[astrocyte_index]
        soma_radius = cell_data.astrocyte_radii[astrocyte_index]

        microdomain = microdomains[astrocyte_index]

    L.debug('Index: {}, Name: {}, Pos: {}, Rad: {}'.format(astrocyte_index, cell_name, soma_position, soma_radius))
    return cell_name, soma_position, soma_radius, microdomain


def obtain_synapse_data(astrocyte_index, synaptic_data_filepath, neuroglial_conn_filepath):
    """ Obtain the synapse coordinates that correspond to the microdomain
    of astrocyte_index
    """

    with \
        SynapticData(synaptic_data_filepath) as synaptic_data, \
        NeuroglialConnectivity(neuroglial_conn_filepath) as neuroglial_connectivity:

        sorted_synapse_idx = sorted(neuroglial_connectivity.astrocyte.to_synapse(astrocyte_index))
        synapse_positions = synaptic_data.synapse_coordinates[sorted_synapse_idx]

    L.debug('Number of synapses for astro index {}: {}'.format(astrocyte_index, len(synapse_positions)))
    return synapse_positions

