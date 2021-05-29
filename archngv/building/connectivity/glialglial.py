"""Generation of connectivity of glial with their neighbors via the formation of gap junctions."""
import logging
import numpy as np
import pandas as pd


L = logging.getLogger(__name__)


# see: https://bbpgitlab.epfl.ch/hpc/touchdetector/-/blob/main/touchdetector/FileWriter.h#L61
BRANCH_MASK = 0xF
BRANCH_SHIFT = 4


def _unpack_types(branch_types):
    """Unpack touchdetector's packed types"""
    return{
        'efferent_section_type': (branch_types >> BRANCH_SHIFT) & BRANCH_MASK,
        'afferent_section_type': branch_types & BRANCH_MASK
    }


def generate_glialglial(touches_directory):
    """Create glial glial connectivity dataframe from touches."""
    from pytouchreader import TouchInfo  # pylint: disable=import-error

    touches = TouchInfo(touches_directory).touches

    # touch names to column names
    touch_props = [
        ("pre_ids", ["source_node_id", "efferent_section_id", "efferent_segment_id"]),
        ("post_ids", ["target_node_id", "afferent_section_id", "afferent_segment_id"]),
        ("distances", ["soma_distance", "efferent_segment_offset", "afferent_segment_offset"]),
        ("pre_section_fraction", ["efferent_section_pos"]),
        ("post_section_fraction", ["afferent_section_pos"]),
        ("spine_length", ["spine_length"]),
        ("pre_position", ["efferent_center_x", "efferent_center_y", "efferent_center_z"]),
        ("post_position", ["afferent_surface_x", "afferent_surface_y", "afferent_surface_z"]),
        ("branch_type", ["branch_type"])
    ]

    properties = {}
    for name, columns in touch_props:
        data = touches[name].to_nparray()

        if len(columns) == 1:
            properties[columns[0]] = data
        else:
            # if there are no touches we want to maintain the shape so that
            # a sonata file is created, albeit empty
            if len(data) == 0:
                for i, column in enumerate(columns):
                    properties[column] = np.empty(0, dtype=data.dtype)
            else:
                for i, column in enumerate(columns):
                    properties[column] = data[:, i]

    # convert branch type into efferent and afferent section types by unpacking it
    properties.update(_unpack_types(properties['branch_type']))

    # chemical synapses do not have soma_distance, therefore we drop it as well for consistency
    for key in ['soma_distance', 'branch_type']:
        del properties[key]

    # create a dataframe and sort it with respect to target_node_id
    return pd.DataFrame(properties).sort_values("target_node_id").reset_index(drop=True)
