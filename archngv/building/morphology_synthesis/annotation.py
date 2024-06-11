# SPDX-License-Identifier: Apache-2.0

""" Annotations for synapses and endfeet surface targets
"""
import morphio
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from archngv.building.types import ASTROCYTE_TO_NEURON
from archngv.exceptions import NGVError

MORPHIO_MAP = {
    "soma": morphio.SectionType.soma,
    "axon": morphio.SectionType.axon,
    "basal": morphio.SectionType.basal_dendrite,
    "apical": morphio.SectionType.apical_dendrite,
}


def _normalized_midpoint_path_distances(segment_begs, segment_ends, segment_offsets):
    """Returns the path distances from the start of the section to each segment
    midpoint normalized by length in the [0, 1]

    Args:
        segment_begs (np.ndarray): Starting points of segments
        segment_ends (np.ndarray): Ending points of segments
        segment_offsets (np.ndarray): The midpoint offset for each segment

    Returns:
        normalized_path_distances (np.ndarray)
    """
    segment_lengths = np.linalg.norm(segment_ends - segment_begs, axis=1)

    path_distances = np.cumsum(segment_lengths)
    total_length = path_distances[-1]
    # shift elements to the right to remove total length and make space for 0.0
    # as first path distance
    path_distances[1:] = path_distances[:-1]
    path_distances[0] = 0

    return (path_distances + segment_offsets) / total_length


def _morphology_unwrapped(morphology):
    """Unwrap a MorphIO morphology into points and their
    respective section id they belong too.

    Args:
        filepath: string

    Returns:
        Tuple of two elements:
            - N x 3 NumPy array with segment midpoints
            - N-row Pandas DataFrame with section ID / segment ID / segment midpoint offset
    """
    data = []

    for section in morphology.iter():
        ps = section.points
        p0s, p1s = ps[:-1], ps[1:]
        midpoints = 0.5 * (p0s + p1s)
        offsets = np.linalg.norm(midpoints - p0s, axis=1)

        normalized_path_distances = _normalized_midpoint_path_distances(p0s, p1s, offsets)

        for segment_id, (midpoint, offset, pdist) in enumerate(
            zip(midpoints, offsets, normalized_path_distances)
        ):
            data.append(
                (midpoint[0], midpoint[1], midpoint[2], section.id, segment_id, offset, pdist)
            )

    if not data:
        raise NGVError("Morphology failed to be unwrapped. There are no points.")

    return pd.DataFrame(
        data,
        columns=["x", "y", "z", "section_id", "segment_id", "segment_offset", "section_position"],
    )


def _endfoot_termination_filter(sections):
    """Checks if a section has the endfoot type and is a termination section,
    which means it has no children
    """
    endfoot_t = MORPHIO_MAP[ASTROCYTE_TO_NEURON["endfoot"]]
    filter_func = lambda section: section.type == endfoot_t and not section.children
    return filter(filter_func, sections)


def annotate_endfoot_location(morphology, endfoot_points):
    """Load a morphology in MorphIO and find the closest point, section
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
    """Load a morphology in MorphIO and find the closest point, section
    to each synapse point.

    Args:
        filepath: string
            Morphology filepath
        synapse_points: float[N, 3]
            Coordinates of the synapses

    Returns:
        Pandas DataFrame with section ID / segment ID / segment offset of closest
        astrocyte segment midpoint.
    """
    data = _morphology_unwrapped(morphology)

    points = data[["x", "y", "z"]].to_numpy()

    _, idx = cKDTree(points, copy_data=False).query(synapse_points)  # pylint: disable=not-callable
    return data.iloc[idx]
