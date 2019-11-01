""" Annotations for synapses and endfeet surface targets
"""
import os
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

import morphio
from archngv import CellData

from archngv.building.morphology_synthesis.endfoot_compartment import create_endfeet_compartment_data
from archngv.building.morphology_synthesis.data_extraction import obtain_endfeet_data
from archngv.building.morphology_synthesis.data_extraction import obtain_synapse_data
from archngv.building.types import ASTROCYTE_TO_NEURON
from archngv.exceptions import NGVError


MORPHIO_MAP = {
    'soma': morphio.SectionType.soma,
    'axon': morphio.SectionType.axon,
    'basal': morphio.SectionType.basal_dendrite,
    'apical': morphio.SectionType.apical_dendrite
}


def _morphology_unwrapped(morphology):
    """ Unwrap a MorphIO morphology into points and their
    respective section id they belong too.

    Args:
        filepath: string

    Returns:
        Tuple of two elements:
            - N x 3 NumPy array with segment midpoints
            - N-row Pandas DataFrame with section ID / segment ID / segment midpoint offset
    """
    points, locations = [], []

    for section in morphology.iter():

        ps = section.points
        p0s, p1s = ps[:-1], ps[1:]
        midpoints = 0.5 * (p0s + p1s)
        offsets = np.linalg.norm(midpoints - p0s, axis=1)

        for segment_id, (midpoint, offset) in enumerate(zip(midpoints, offsets)):
            points.append(midpoint)
            locations.append((section.id, segment_id, offset))

    if not points:
        raise NGVError("Morphology failed to be unwrapped. There are no points.")

    return np.asarray(points), pd.DataFrame(locations, columns=['section_id', 'segment_id', 'segment_offset'])


def _endfoot_termination_filter(sections):
    """ Checks if a sction has the endfoot type and is a termination section,
        which means it has no children
    """
    endfoot_t = MORPHIO_MAP[ASTROCYTE_TO_NEURON['endfoot']]
    filter_func = lambda section: section.type == endfoot_t and not section.children
    return filter(filter_func, sections)


def annotate_endfoot_location(morphology, endfoot_points):
    """ Load a morphology in MorphIO and find the closest point, section
    to each endfoot point.

    Args:
        filepath: string
            Morphology filepath
        endfoot_points: float[N, 3]
            Coordinates of the endfeet touch points

    Returns:
        section_ids: array[int, (len(endfoot_points),)]
    """
    points, section_ids = [], []
    for section in _endfoot_termination_filter(morphology.iter()):
        points.append(section.points[-1])
        section_ids.append(section.id)

    points, section_ids = np.asarray(points), np.asarray(section_ids)
    _, idx = cKDTree(points, copy_data=False).query(endfoot_points)  # pylint: disable=not-callable
    return section_ids[idx]


def annotate_synapse_location(morphology, synapse_points):
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
    points, locations = _morphology_unwrapped(morphology)
    _, idx = cKDTree(points, copy_data=False).query(synapse_points)  # pylint: disable=not-callable
    return locations.iloc[idx]


def create_astrocyte_annotations(astrocyte_index, paths):
    """ Generates annotations for endfeet and synapses, endfoot compartment information
    and astrocyte section perimeters.
    """
    cell_data = CellData(paths.cell_data)
    cell_name = str(cell_data.astrocyte_names[astrocyte_index], 'utf-8')

    # For the annotations the morphology should be in readonly mode. If it is mutated for any reason
    # there will be reordering of the sections ids and thus the annotations would be invalid
    morph_filepath = os.path.join(paths.morphology_directory, cell_name + '.h5')
    morphology = morphio.Morphology(morph_filepath, options=morphio.Option.nrn_order)

    # pylint: disable=too-many-arguments
    annotations = {}

    # get endfeet targets on vasculature surface
    endfeet_data = obtain_endfeet_data(astrocyte_index,
                                       paths.gliovascular_data,
                                       paths.gliovascular_connectivity,
                                       paths.endfeet_areas)

    if endfeet_data is not None:
        annotations['endfeet'] = endfeet_dict = {}
        endfeet_dict['morph_section_ids'] = annotate_endfoot_location(morphology, endfeet_data.targets)
        endfeet_dict['compartments'] = create_endfeet_compartment_data(morphology, endfeet_data)

    # extract synapses coordinates
    synapses = obtain_synapse_data(astrocyte_index, paths.synaptic_data, paths.neuroglial_connectivity)

    if synapses is not None:
        annotations['synapses'] = syn_dict = {}
        syn_dict['synapse_ids'] = synapses.index.values

        locs = annotate_synapse_location(morphology, synapses.values)

        syn_dict['morph_locations'] = (locs.section_id.to_numpy(),
                                       locs.segment_id.to_numpy(),
                                       locs.segment_offset.to_numpy())

    return annotations
