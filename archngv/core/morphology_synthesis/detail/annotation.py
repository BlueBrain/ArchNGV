
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


def _morphology_unwrapped(filepath, filter_func=None):

    morphology = morphio.Morphology(filepath)

    root_sections = \
    morphology.root_sections if filter_func is None else filter(filter_func, morphology.root_sections)

    points_section_idx = []
    points = []

    for root_section in root_sections:
        for section in root_section.iter():

            section_id = section.id
            for point in section.points:

                points_section_idx.append(section_id)
                points.append(point)

    points = np.asarray(points)

    return points_section_idx, points


def annotate_endfoot_location(filepath, endfoot_points):

    endfoot_type = MORPHIO_MAP[MAP_TO_NEURONAL['endfoot']]

    points_section_idx, points = _morphology_unwrapped(filepath, lambda s: s.type == endfoot_type)

    _, idx = cKDTree(points, copy_data=False).query(endfoot_points)

    endfeet_location = [(points_section_idx[index], index) for index in idx]

    return endfeet_location


def export_endfoot_location(filepath, endfoot_points):

    endfeet_location = annotate_endfoot_location(filepath, endfoot_points)

    data_filepath = filepath.replace('.h5', '_endfeet_annotation.h5')

    with h5py.File(data_filepath, 'w') as fd:
        fd.create_dataset('endfeet_location', data=endfeet_location, dtype=np.intp)


def annotate_synapse_location(filepath, synapse_points):

    points_section_idx, points = _morphology_unwrapped(filepath, None)

    _, idx = cKDTree(points, copy_data=False).query(synapse_points)

    synapse_location = [(points_section_idx[index], index) for index in idx]

    return synapse_location


def export_synapse_location(filepath, synapse_points):

    synapse_location = annotate_synapse_location(filepath, synapse_points)

    data_filepath = filepath.replace('.h5', '_synapse_annotation.h5')
    with h5py.File(data_filepath, 'w') as fd:
        fd.create_dataset('synapse_location', data=synapse_location, dtype=np.intp)

