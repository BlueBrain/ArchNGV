""" Data extraction for synthesis workers
"""

import logging
from collections import namedtuple

import pandas as pd

from archngv.core.data_cells import CellData
from archngv.core.data_synaptic import SynapticData
from archngv.core.data_endfeet_areas import EndfeetAreas
from archngv.core.data_gliovascular import GliovascularData
from archngv.core.data_microdomains import MicrodomainTesselation
from archngv.core.connectivity_neuroglial import NeuroglialConnectivity
from archngv.core.connectivity_gliovascular import GliovascularConnectivity

L = logging.getLogger(__name__)


EndfeetData = namedtuple('EndfeetData', ['targets', 'area_meshes'])


def obtain_endfeet_data(astrocyte_index,
                        gliovascular_data_path,
                        gliovascular_connectivity_path,
                        endfeet_areas_path):
    """ Extract the endfeet information from astrocyte_index if any, otherwise return None
    """
    gliovascular_connectivity = GliovascularConnectivity(gliovascular_connectivity_path)
    endfeet_indices = gliovascular_connectivity.astrocyte.to_endfoot(astrocyte_index)

    if len(endfeet_indices) == 0:
        L.warning('No endfeet found for astrocyte index %d', astrocyte_index)
        return None

    gliovascular_data = GliovascularData(gliovascular_data_path)
    targets = gliovascular_data.endfoot_surface_coordinates[sorted(endfeet_indices)]

    endfeet_areas = EndfeetAreas(endfeet_areas_path)[endfeet_indices]

    L.debug('Found endfeet %s for astrocyte index %d', endfeet_indices, astrocyte_index)
    L.debug('Endfeet Coordinates: %s', targets)
    L.debug('Endfeet Area Meshes: %s', endfeet_areas)

    return EndfeetData(targets=targets, area_meshes=endfeet_areas)


def obtain_cell_properties(astrocyte_index,
                           cell_data_filepath,
                           microdomains_filepath):
    """ Extract the cell info (cell_name, pos and radius) and its microdomain
    via the ngv_config and its index.
    """

    microdomains = MicrodomainTesselation(microdomains_filepath)

    with CellData(cell_data_filepath) as cell_data:

        cell_name = str(cell_data.astrocyte_names[astrocyte_index], 'utf-8')
        soma_position = cell_data.astrocyte_positions[astrocyte_index]
        soma_radius = cell_data.astrocyte_radii[astrocyte_index]

        microdomain = microdomains[astrocyte_index]

    L.debug('Index: %d, Name: %s, Pos: %s, Rad: %f', astrocyte_index, cell_name, soma_position, soma_radius)
    return cell_name, soma_position, soma_radius, microdomain


def obtain_synapse_data(astrocyte_index, synaptic_data_filepath, neuroglial_conn_filepath):
    """ Obtain the synapse coordinates that correspond to the microdomain
    of astrocyte_index
    """

    with SynapticData(synaptic_data_filepath) as synaptic_data, \
         NeuroglialConnectivity(neuroglial_conn_filepath) as neuroglial_connectivity:

        synapse_ids = sorted(neuroglial_connectivity.astrocyte_synapses(astrocyte_index))

        if len(synapse_ids) == 0:
            L.warning('No synapses found for astrocyte index %d', astrocyte_index)
            return None

        positions = synaptic_data.synapse_coordinates(synapse_ids)

    L.debug('Number of synapses for astro index %d: %d', astrocyte_index, len(positions))

    return pd.DataFrame(index=synapse_ids, data=positions, columns=['x', 'y', 'z'])
