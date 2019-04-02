""" Facade classes for NGV connectivity
"""
import time
import logging

import numpy

from spatial_index import point_rtree
from morphspatial.collision import convex_shape_with_spheres

from ..data_structures.data_synaptic import SynapticData
from ..data_structures.data_microdomains import MicrodomainTesselation
from ..data_structures.connectivity_synaptic import SynapticConnectivity


L = logging.getLogger(__name__)


def generate_neuroglial(n_astrocytes, ngv_config, map_func):  # pylint: disable = unused-argument
    """ Yields the connectivity of the i-th astrocyte with synapses and neurons
    """
    with MicrodomainTesselation(ngv_config.output_paths('overlapping_microdomain_structure')) as mdom, \
         SynapticData(ngv_config.output_paths('synaptic_data')) as syn_data, \
         SynapticConnectivity(ngv_config.output_paths('synaptic_connectivity')) as syn_conn:

        synapse_coordinates = syn_data.synapse_coordinates[:]
        synapse2neuron = syn_conn.synapse.to_afferent_neuron_map[:]

        index = point_rtree(synapse_coordinates)

        for astrocyte_index in range(n_astrocytes):

            start_time = time.time()

            domain = mdom.domain_object(astrocyte_index)

            masked_idx = spheres_inside_domain(index, synapse_coordinates, domain)
            neuron_indices = numpy.unique(synapse2neuron[masked_idx])

            yield {'domain_synapses': masked_idx,
                   'domain_neurons': neuron_indices}

            elapsed_time = time.time() - start_time
            L.info('Index: %s, Ninside: %s, Nneurons: %s, ET: %s', astrocyte_index,
                                                                   len(masked_idx),
                                                                   len(neuron_indices),
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
