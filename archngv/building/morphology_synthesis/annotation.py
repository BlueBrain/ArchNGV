""" Annotations for synapses and endfeet surface targets
"""

import h5py
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

import morphio
from archngv.building.types import ASTROCYTE_TO_NEURON
from archngv.exceptions import NGVError

MORPHIO_MAP = {
    'soma': morphio.SectionType.soma,
    'axon': morphio.SectionType.axon,
    'basal': morphio.SectionType.basal_dendrite,
    'apical': morphio.SectionType.apical_dendrite
}


def _morphology_unwrapped(morphology, filter_func=None):
    """ Unwrap a MorphIO morphology into points and their
    respective section id they belong too.

    Args:
        filepath: string

    Returns:
        Tuple of two elements:
            - N x 3 NumPy array with segment midpoints
            - N-row Pandas DataFrame with section ID / segment ID / segment midpoint offset
    """
    section_iterator = morphology.iter()

    if filter_func is not None:
        section_iterator = filter(filter_func, section_iterator)

    points = []
    locations = []

    for section in section_iterator:
        p0s = section.points[:-1]
        p1s = section.points[1:]
        midpoints = 0.5 * (p0s + p1s)
        offsets = np.linalg.norm(midpoints - p0s, axis=1)
        for segment_id, (midpoint, offset) in enumerate(zip(midpoints, offsets)):
            points.append(midpoint)
            locations.append((section.id, segment_id, offset))

    if not points:
        raise NGVError("Morphology failed to be unwrapped.")

    points = np.asarray(points)
    locations = pd.DataFrame(locations, columns=['section_id', 'segment_id', 'segment_offset'])

    return points, locations


def _is_endfoot_termination(section):
    """ Checks if a sction has the endfoot type and is a termination section,
        which means it has no children
    """
    endfoot_t = MORPHIO_MAP[ASTROCYTE_TO_NEURON['endfoot']]
    return section.type == endfoot_t and not section.children


def annotate_endfoot_location(mutable_morphology, endfoot_points):
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
    points, locations = _morphology_unwrapped(mutable_morphology, filter_func=_is_endfoot_termination)

    _, idx = cKDTree(points, copy_data=False).query(endfoot_points)  # pylint: disable=not-callable

    return locations.loc[idx]


def export_endfoot_location(filepath, endfeet_annotation):
    """ Calculate the endfeet annotations and export them to filen
    """
    data_filepath = filepath.replace('.h5', '_endfeet_annotation.h5')

    with h5py.File(data_filepath, 'w') as fd:
        root = fd.create_group('endfeet_location')
        for prop, column in endfeet_annotation.iteritems():
            root[prop] = column.values


def annotate_synapse_location(mutable_morphology, synapse_points):
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
    points, locations = _morphology_unwrapped(mutable_morphology)

    _, idx = cKDTree(points, copy_data=False).query(synapse_points)  # pylint: disable=not-callable

    return locations.loc[idx]


def export_synapse_location(filepath, synapses, synapses_location):
    """ Calculate the synapses annotation and export them to file
    """
    data_filepath = filepath.replace('.h5', '_synapse_annotation.h5')
    with h5py.File(data_filepath, 'w') as fd:
        fd['synapse_id'] = synapses.index
        root = fd.create_group('synapse_location')
        for prop, column in synapses_location.iteritems():
            root[prop] = column.values
