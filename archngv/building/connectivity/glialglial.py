"""Generation of connectivity of glial with their neighbors via the formation of gap junctions."""
import logging
import numpy as np
import pandas as pd


L = logging.getLogger(__name__)


def generate_glialglial(touches_directory):
    """Create glial glial connectivity dataframe from touches."""
    from pytouchreader import TouchInfo  # pylint: disable=import-error

    touches = TouchInfo(touches_directory).touches

    # touch names to column names
    touch_props = [
        ("pre_ids", ["pre_id", "pre_section_id", "pre_segment_id"]),
        ("post_ids", ["post_id", "post_section_id", "post_segment_id"]),
        ("distances", ["distances_x", "distances_y", "distances_z"]),
        ("pre_section_fraction", ["pre_section_fraction"]),
        ("post_section_fraction", ["post_section_fraction"]),
        ("spine_length", ["spine_length"]),
        ("pre_position", ["efferent_center_x", "efferent_center_y", "efferent_center_z"]),
        ("post_position", ["afferent_surface_x", "afferent_surface_y", "afferent_surface_z"]),
        ("branch_type", ["branch_type"])
    ]

    properties = pd.DataFrame(index=np.arange(len(touches)))
    for touch_name, df_columns in touch_props:
        properties = properties.join(pd.DataFrame(data=touches[touch_name].to_nparray(),
                                                  columns=df_columns, index=properties.index))

    return properties.sort_values("post_id").reset_index(drop=True)
