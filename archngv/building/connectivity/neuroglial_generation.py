""" Neuroglial Connectivity
"""

import logging
from builtins import range

import numpy as np
import pandas as pd
from spatial_index import SphereIndex

from archngv.spatial.collision import convex_shape_with_spheres

L = logging.getLogger(__name__)


def spheres_inside_domain(index, synapse_coordinates, domain):
    """
    Returns the indices of the spheres that are inside
    the convex geometry
    """
    query_window = domain.bounding_box

    idx = index.find_intersecting_window(query_window[:3], query_window[3:])
    mask = convex_shape_with_spheres(
        domain.face_points,
        domain.face_normals,
        synapse_coordinates[idx],
        np.zeros(len(idx)),
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


def generate_neuroglial(astrocytes, microdomains, neuronal_connectivity):
    """Yields the connectivity of the astrocyte ids with synapses and neurons

    Args:
        astrocytes: voxcell.NodePopulation
        microdomains: MicrodomainTesselation
        neuronal_connectivity: NeuronalConnectivity

    Returns:
        DataFrame with 'astrocyte_id', 'synapse_id', 'neuron_id'
    """
    synapse_coordinates = neuronal_connectivity.synapse_positions()
    synapse_to_neuron = neuronal_connectivity.target_neurons()

    index = SphereIndex(synapse_coordinates, radii=None)

    ret = []
    for astrocyte_id in range(len(astrocytes.properties)):
        domain = microdomains[astrocyte_id]
        synapses_ids = astrocyte_neuroglial_connectivity(domain, index, synapse_coordinates)
        ret.append(
            pd.DataFrame(
                {
                    "astrocyte_id": astrocyte_id,
                    "synapse_id": synapses_ids,
                    "neuron_id": synapse_to_neuron[synapses_ids],
                }
            )
        )

    ret = pd.concat(ret)
    ret.sort_values(["neuron_id", "astrocyte_id", "synapse_id"], inplace=True)
    return ret
