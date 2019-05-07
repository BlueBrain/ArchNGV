""" Annotations for synapses and endfeet surface targets
"""
import morphio

import h5py
import numpy as np
from scipy.spatial import cKDTree

from .morphology_types import MAP_TO_NEURONAL


MORPHIO_MAP = \
    {
         'soma'   : morphio.SectionType.soma,
         'axon'   : morphio.SectionType.axon,
         'basal'  : morphio.SectionType.basal_dendrite,
         'apical' : morphio.SectionType.apical_dendrite
    }


def _morphology_unwrapped(filepath, neurite_filter=lambda s: True):
    """ Unwrap a MorphIO morphology into points and their
    respective section id they belong too.

    Args:
        filepath: string

    Returns:
        Tuple of two lists:
            - section ids of morphology points
            - coordinates of morphology points
    """
    morphology = morphio.Morphology(filepath)

    points_section_idx = []
    points = []

    for root_section in filter(neurite_filter, morphology.root_sections):
        for section in root_section.iter():

            section_id = section.id
            for point in section.points:

                points_section_idx.append(section_id)
                points.append(point)

    points = np.asarray(points)

    return points_section_idx, points


def annotate_endfoot_location(filepath, endfoot_points):
    """ Load a morphology in MorphIO and find the closest point, section
    to each endfoot point.

    Args:
        filepath: string
            Morphology filepath
        endfoot_points: float[N, 3]
            Coordinates of the endfeet touch points

    Returns:
        A list of tuples, where the i-th entry corresponds to
        the i-th row in the endfoot_points and each tuple contains:
            - Closest section id
            - Closest point index
    """
    endfoot_type = MORPHIO_MAP[MAP_TO_NEURONAL['endfoot']]

    points_section_idx, points = _morphology_unwrapped(filepath, neurite_filter=lambda s: s.type == endfoot_type)

    _, idx = cKDTree(points, copy_data=False).query(endfoot_points)

    endfeet_annotation = [(points_section_idx[index], index) for index in idx]

    return endfeet_annotation


def export_endfoot_location(filepath, endfoot_points):
    """ Calculate the endfeet annotations and export them to filen
    """
    endfeet_annotation = annotate_endfoot_location(filepath, endfoot_points)

    data_filepath = filepath.replace('.h5', '_endfeet_annotation.h5')

    with h5py.File(data_filepath, 'w') as fd:
        fd.create_dataset('endfeet_location', data=endfeet_annotation, dtype=np.intp)


def annotate_synapse_location(filepath, synapse_points):
    """ Load a morphology in MorphIO and find the closest point, section
    to each synapse point.

    Args:
        filepath: string
            Morphology filepath
        synapse_points: float[N, 3]
            Coordinates of the synapses

    Returns:
        A list of tuples, where the i-th entry corresponds to
        the i-th row in the endfoot_points and each tuple contains:
            - Closest section id
            - Closest point index
    """
    points_section_idx, points = _morphology_unwrapped(filepath)

    _, idx = cKDTree(points, copy_data=False).query(synapse_points)

    synapse_annotation = [(points_section_idx[index], index) for index in idx]

    return synapse_annotation


def export_synapse_location(filepath, synapse_points):
    """ Calculate the synapses annotation and export them to file
    """
    synapse_location = annotate_synapse_location(filepath, synapse_points)

    data_filepath = filepath.replace('.h5', '_synapse_annotation.h5')
    with h5py.File(data_filepath, 'w') as fd:
        fd.create_dataset('synapse_location', data=synapse_location, dtype=np.intp)
