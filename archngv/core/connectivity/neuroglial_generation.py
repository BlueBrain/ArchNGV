""" Facade classes for NGV connectivity
"""
import os
import h5py
import numpy
import logging
import numpy as np
from scipy.spatial import cKDTree

import rtree
from spatial_index import point_rtree


from morphspatial import collision
from morphspatial.spatial_index import FastRtree

from ..data_structures.data_microdomains import MicrodomainTesselation
from ..data_structures.data_synaptic import SynapticData
from ..data_structures.connectivity_synaptic import SynapticConnectivity
import time

L = logging.getLogger(__name__)


def generate_neuroglial(n_astrocytes, ngv_config, map_func):

    with MicrodomainTesselation(ngv_config.output_paths('overlapping_microdomain_structure')) as mdom, \
         SynapticData(ngv_config.output_paths('synaptic_data')) as syn_data, \
         SynapticConnectivity(ngv_config.output_paths('synaptic_connectivity')) as syn_conn:

        synapse_coordinates = syn_data.synapse_coordinates[:]
        synapse2neuron = syn_conn.synapse.to_afferent_neuron_map[:]

        L.info("Generating spatial index from synapse positions")
        index = point_rtree(synapse_coordinates)

        for astrocyte_index in range(n_astrocytes):

            domain = mdom.domain_object(astrocyte_index)
            query_window = tuple(domain.bounding_box)

            t0 = time.time()

            idx = index.intersection(*query_window)
            coordinates = synapse_coordinates[idx]

            mask = collision.convex_shape_with_spheres(
                                                           domain.face_points,
                                                           domain.face_normals,
                                                           coordinates,
                                                           numpy.zeros(len(coordinates))
                                                      )
            masked_idx = idx[mask]
            neuron_indices = np.unique(synapse2neuron[masked_idx])


            yield {'domain_synapses': masked_idx,
                   'domain_neurons': neuron_indices}

            L.info('Index: {}, Npoints: {}, Ninside: {}, Nneurons: {}, Elapsed Time: {}'.format(astrocyte_index, len(idx), len(masked_idx), len(neuron_indices), time.time() - t0))


def generate_neuroglial_bak(n_astrocytes, ngv_config, map_func):

    L.info('Started')

    synapses_index_path = ngv_config.output_paths('synapses_index')

    # get the spatial index
    properties = rtree.index.Property()
    properties.dimension = 3
    index = FastRtree(synapses_index_path, properties=properties)

    L.info('Loaded Index')

    with MicrodomainTesselation(ngv_config.output_paths('overlapping_microdomain_structure')) as mdom, \
         SynapticData(ngv_config.output_paths('synaptic_data')) as syn_data, \
         SynapticConnectivity(ngv_config.output_paths('synaptic_connectivity')) as syn_conn:

        synapse_coordinates = syn_data.synapse_coordinates[:]

        synapse2neuron = syn_conn.synapse.to_afferent_neuron_map[:]

        for astrocyte_index in range(n_astrocytes):

            domain = mdom.domain_object(astrocyte_index)
            query_window = tuple(domain.bounding_box)

            t0 = time.time()

            idx = np.fromiter(index.intersection(query_window), dtype=np.uintp)
            idx.sort()


            coordinates = synapse_coordinates[idx]

            mask = collision.convex_shape_with_spheres(
                                                           domain.face_points,
                                                           domain.face_normals,
                                                           coordinates,
                                                           numpy.zeros(len(coordinates))
                                                      )

            masked_idx = idx[mask]
            neuron_indices = np.unique(synapse2neuron[masked_idx])


            yield {'domain_synapses': masked_idx,
                   'domain_neurons': neuron_indices}

            L.debug('Index: {}, Npoints: {}, Ninside: {}, Nneurons: {}, Elapsed Time: {}'.format(astrocyte_index, len(idx), len(masked_idx), len(neuron_indices), time.time() - t0))

