# SPDX-License-Identifier: Apache-2.0

"""
Functions related to grouping
"""
import numpy as np


class GroupedElements:
    """Class for easier access to grouped elements"""

    def __init__(self, ids, offsets, groups):
        self.ids = ids
        self._offsets = offsets
        self.groups = groups

    def items(self):
        """
        Yields:
            group,
            group_ids
        """
        for group_index, group in enumerate(self.groups):
            yield group, self.get_group_ids(group_index)

    def iter_assigned_groups(self):
        """
        Returns iterator of the groups that are not -1 and their respective
        triangle ids
        """
        for group, ids in self.items():
            if group >= 0:
                yield group, ids

    def get_group_ids(self, group_index):
        """Returns the triangle ids of group_index"""
        return self.ids[self._offsets[group_index] : self._offsets[group_index + 1]]


def group_elements(v_group_index):
    """transform v_group_index (storing vertex -> group information) into inverse

                   0 1 2 3 4 5 6  7  8  # (implicit vertex index)
    v_group_index  3 3 3 2 2 1 0 -1 -1  # group; -1 means not specified
    ->
              0  1  2  3  4  5  6  # index
       idx = [6, 5, 3, 4, 0, 1, 2, ]
                  0  1  2  3      # implicit group index
       offsets = [0, 1, 2, 4, 7]  # offsets into above
    """
    # group vertices with same seed, note: casting to uint to have -1 sort at the end
    values = v_group_index

    idx = np.argsort(values, kind="stable")

    # find unique groups in values and their respective offsets
    groups, offsets = np.unique(values, return_counts=True)

    cumulative_offsets = np.empty(len(offsets) + 1, dtype=np.int64)

    cumulative_offsets[0] = 0
    cumulative_offsets[1:] = np.cumsum(offsets)

    return GroupedElements(idx, cumulative_offsets, groups)


def vertex_to_triangle_groups(vertex_groups, triangles):
    """
        Maps from the vertex groups to triangle groups.

    Args:
        vertex_groups:
            Group for each vertex. Can also be unassigned -1
        triangles:
            Mesh triangles

    Returns:
        The groups of triangles. They can also be unassigned -1.

    Note:
        A triangle will acquire a group if all three vertices have the same group. Otherwise it
        becomes unassigned -1.
    """
    # triangles with vertex groups instead of vertex indices
    group_tris = vertex_groups[triangles]

    triangle_groups = np.full(len(triangles), fill_value=-1, dtype=np.int64)

    # first second cols equal or first third cols equal
    mask = (group_tris[:, 0] == group_tris[:, 1]) | (group_tris[:, 0] == group_tris[:, 2])
    triangle_groups[mask] = group_tris[mask, 0]

    # second third cols equal
    mask = group_tris[:, 1] == group_tris[:, 2]
    triangle_groups[mask] = group_tris[mask, 1]

    return triangle_groups
