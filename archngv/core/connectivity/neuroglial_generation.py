""" Neuroglial Connectivity
"""

from builtins import range

import time
import logging

import numpy as np
import pandas as pd

from spatial_index import point_rtree
from morphspatial.collision import convex_shape_with_spheres


L = logging.getLogger(__name__)


def spheres_inside_domain(index, synapse_coordinates, domain):
    """
        Returns the indices of the spheres that are inside
        the convex geometry
    """
    query_window = tuple(domain.bounding_box)

    idx = index.intersection(*query_window)
    mask = convex_shape_with_spheres(
                                       domain.face_points,
                                       domain.face_normals,
                                       synapse_coordinates[idx],
                                       np.zeros(len(idx))
    )
    return idx[mask]


def astrocyte_neuroglial_connectivity(microdomain, synapses_spatial_index, synapse_coordinates):
    """
    Args:
        microdomain: ConvexPolygon
        synapses_spatial_index: point_rtree

    Returns:
        synapses_ids: array[int, (M,)]

        The M synapses ids that lie inside microdomain geometry and their respective neuron ids.
    """
    return spheres_inside_domain(synapses_spatial_index, synapse_coordinates, microdomain)


def generate_neuroglial(astrocytes, microdomains, synaptic_data):
    """ Yields the connectivity of the astrocyte ids with synapses and neurons

    Args:
        astrocytes: voxcell.NodePopulation
        microdomains: MicrodomainTesselation
        synaptic_data: SynapticData

    Yields:
        (astrocyte_id, synapses) pair, where `synapses` is pandas DataFrame with columns:
            - 'synapse_id' (as seen in the `synaptic_data`)
            - 'neuron_id' (postsynaptic neuron GID)
    """
    synapse_coordinates = synaptic_data.synapse_coordinates()
    synapse_to_neuron = synaptic_data.afferent_gids()

    index = point_rtree(synapse_coordinates)

    for astrocyte_id in range(astrocytes.size):

        domain = microdomains[astrocyte_id]

        start_time = time.time()

        synapses_ids = \
            astrocyte_neuroglial_connectivity(domain, index, synapse_coordinates)

        synapses = pd.DataFrame({
            'synapse_id': synapses_ids,
            'neuron_id': synapse_to_neuron[synapses_ids],
        })

        yield astrocyte_id, synapses

        elapsed_time = time.time() - start_time
        L.info('Index: %s, Ninside: %s, Nneurons: %s, ET: %s', astrocyte_id,
                                                               len(synapses),
                                                               len(synapses['neuron_id'].unique()),
                                                               elapsed_time)
