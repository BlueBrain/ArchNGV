""" Annotations for synapses and endfeet surface targets
"""
import morphio

import h5py
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

from archngv.core.types import ASTROCYTE_TO_NEURON


MORPHIO_MAP = {
    'soma': morphio.SectionType.soma,
    'axon': morphio.SectionType.axon,
    'basal': morphio.SectionType.basal_dendrite,
    'apical': morphio.SectionType.apical_dendrite
}


def _morphology_unwrapped(filepath, neurite_filter=lambda s: True):
    """ Unwrap a MorphIO morphology into points and their
    respective section id they belong too.

    Args:
        filepath: string

    Returns:
        Tuple of two elements:
            - N x 3 NumPy array with segment midpoints
            - N-row Pandas DataFrame with section ID / segment ID / segment midpoint offset
    """
    morphology = morphio.Morphology(filepath, options=morphio.Option.nrn_order)

    points = []
    locations = []

    for root_section in filter(neurite_filter, morphology.root_sections):
        for section in root_section.iter():
            p0 = section.points[:-1]
            p1 = section.points[1:]
            midpoints = 0.5 * (p0 + p1)
            offsets = np.linalg.norm(midpoints - p0, axis=1)
            for segment_id, (midpoint, offset) in enumerate(zip(midpoints, offsets)):
                points.append(midpoint)
                locations.append((section.id, segment_id, offset))

    points = np.asarray(points)
    locations = pd.DataFrame(locations, columns=['section_id', 'segment_id', 'segment_offset'])

    return points, locations


def annotate_endfoot_location(filepath, endfoot_points):
    """ Load a morphology in MorphIO and find the closest point, section
    to each endfoot point.

    Args:
        filepath: string
            Morphology filepath
        endfoot_points: float[N, 3]
            Coordinates of the endfeet touch points

    Returns:
        Pandas DataFrame with section ID / segment ID / segment offset of closest astrocyte segment midpoint
    """
    endfoot_type = MORPHIO_MAP[ASTROCYTE_TO_NEURON['endfoot']]

    points, locations = _morphology_unwrapped(filepath, neurite_filter=lambda s: s.type == endfoot_type)

    _, idx = cKDTree(points, copy_data=False).query(endfoot_points)

    return locations.loc[idx]


def export_endfoot_location(filepath, endfoot_points):
    """ Calculate the endfeet annotations and export them to filen
    """
    endfeet_annotation = annotate_endfoot_location(filepath, endfoot_points)

    data_filepath = filepath.replace('.h5', '_endfeet_annotation.h5')

    with h5py.File(data_filepath, 'w') as fd:
        root = fd.create_group('endfeet_location')
        for prop, column in endfeet_annotation.iteritems():
            root[prop] = column.values


def annotate_synapse_location(filepath, synapse_points):
    """ Load a morphology in MorphIO and find the closest point, section
    to each synapse point.

    Args:
        filepath: string
            Morphology filepath
        synapse_points: float[N, 3]
            Coordinates of the synapses

    Returns:
        Pandas DataFrame with section ID / segment ID / segment offset of closest astrocyte segment midpoint
    """
    points, locations = _morphology_unwrapped(filepath)

    _, idx = cKDTree(points, copy_data=False).query(synapse_points)

    return locations.loc[idx]


def export_synapse_location(filepath, synapses):
    """ Calculate the synapses annotation and export them to file
    """
    synapse_location = annotate_synapse_location(filepath, synapses.values)

    data_filepath = filepath.replace('.h5', '_synapse_annotation.h5')
    with h5py.File(data_filepath, 'w') as fd:
        fd['synapse_id'] = synapses.index
        root = fd.create_group('synapse_location')
        for prop, column in synapse_location.iteritems():
            root[prop] = column.values
