""" Neuroglial Connectivity
"""
import time
import logging

import numpy

from spatial_index import point_rtree
from morphspatial.collision import convex_shape_with_spheres


L = logging.getLogger(__name__)


def astrocyte_neuroglial_connectivity(microdomain, synapses_spatial_index, synapse_coordinates, synapse_to_neuron):
    """
    Args:
        microdomain: ConvexPolygon
        synapses_spatial_index: point_rtree
        synapse_to_neuron: array[int, (N,)]

    Returns:
        synapses_ids: array[int, (M,)]
        neuron_ids: array[int, (M, )]

        The M synapses ids that lie inside microdomain geometry and their respective neuron ids.
    """
    synapses_ids = spheres_inside_domain(synapses_spatial_index, synapse_coordinates, microdomain)
    neuron_ids = numpy.unique(synapse_to_neuron[synapses_ids])
    return synapses_ids, neuron_ids


def generate_neuroglial(astrocytes, microdomains, synaptic_data):
    """ Yields the connectivity of the astrocyte ids with synapses and neurons

    Args:

        astrocytes: voxcell.NodePopulation
        microdomains: MicrodomainTesselation
        synaptic_data: SynapticData
    """
    synapse_coordinates = synaptic_data.synapse_coordinates()
    synapse_to_neuron = synaptic_data.afferent_gids()

    index = point_rtree(synapse_coordinates)

    for astrocyte_index in numpy.arange(astrocytes.size):

        domain = microdomains[astrocyte_index]

        start_time = time.time()

        synapses_ids, neuron_ids = \
            astrocyte_neuroglial_connectivity(domain, index, synapse_coordinates, synapse_to_neuron)

        yield synapses_ids, neuron_ids

        elapsed_time = time.time() - start_time
        L.info('Index: %s, Ninside: %s, Nneurons: %s, ET: %s', astrocyte_index,
                                                               len(synapses_ids),
                                                               len(neuron_ids),
                                                               elapsed_time)


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
                                       numpy.zeros(len(idx))
    )
    return idx[mask]
