"""
Utilities for the microdomain tesselation data structure
"""
import itertools
import numpy as np

from archngv.utils.geometry import unique_points
from archngv.utils.functional import consecutive_pairs


def local_to_global_triangles(triangles, ps_tris_offsets, local_to_global_vertices):
    """ Converts and array from the local index space to the global one

    Args:
        triangles: array[int, (N, 3)]
        ps_tris_offsets: array[int, (N + 1,)]
        local_to_global_vertices: array[int, (M,)]
    """
    global_tris = np.empty_like(triangles)
    total_tris = 0
    for (p_beg, t_beg), (p_end, t_end) in consecutive_pairs(ps_tris_offsets):

        # get the local triangles for the i-th astrocyte
        local_tris = triangles[t_beg: t_end]
        n_tris = len(local_tris)

        # get the local_to_global vertex map slice for the i-th asotrycte
        l2g = local_to_global_vertices[p_beg: p_end]

        # using the local to global map to transform the local triangle vertices
        # to the global unique ones
        global_tris[total_tris: total_tris + n_tris] = l2g[local_tris]
        total_tris += n_tris

    return global_tris


def local_to_global_mapping(points, triangles, ps_tris_offsets, triangle_labels=None, decimals=4):
    """ Given an array of points return an array of indices that correspond
    to all the unique points in the array.

    1D Example:

    arr = [0.22, 0.11, 0.22, 0.33, 0.11, 0.44, 0.44, 0.44, 0.44, 0.0]
    vertices = [2, 1, 2, 3, 1, 4, 4, 4, 4, 0]

    """
    unique_idx, ps_to_uverts_map = unique_points(points, decimals=decimals)
    global_tris = local_to_global_triangles(triangles, ps_tris_offsets, ps_to_uverts_map)

    # because vertices array has the same unique vertex id for duplicate coordinates
    # when we remapped the triangles we actually mapped to the unique index space.
    # Finally we remove the duplicate triangles via unique across rows after we make
    # sure that all the triangle ids are sorted
    sorted_cols_tris = np.sort(global_tris, axis=1, kind='mergesort')
    _, idx = np.unique(sorted_cols_tris, axis=0, return_index=True)

    # keep the initial order of the triangles
    # when selecting unique rows
    idx.sort(kind='mergesort')

    global_tris = global_tris[idx]

    if triangle_labels is None:
        return points[unique_idx], global_tris
    return points[unique_idx], global_tris, triangle_labels[idx]


def local_to_global_polygon_ids(polygon_ids):
    """ Given an array of increasing polygon_ids stored in the local index space
    Example:

        polygon_ids = [0, 0, 1, 1, 0, 1, 1, 2]
        global_poly = [0, 0, 1, 1, 2, 3, 3, 4]
    """
    is_different = np.empty(polygon_ids.shape, dtype=np.bool)
    is_different[0] = False
    # check if the i and i-1 ids are different
    is_different[1:] = polygon_ids[1:] != polygon_ids[:-1]
    # the cumulative sum trackes the previous values so
    # that the index always increases with a new non duplicate value
    return np.cumsum(is_different)


def triangles_to_polygons(triangles, polygon_ids):
    """ Converts triangles to a polygon list

    Args:
        triangles: array[int, (N, 3)]
            Array of triangles in specific order to reconstruct polygons
        polygon_ids: array[int, N]
            The polygon id that each triangle belongs to.
    Returns:
        polygons: list[list[int]]

    Notes:
        Triangles have to be ordered in the following way:
            [0, 1, 2],
            [0, 2, 3],
            [0, 2, 4]
        so that the polygon can right away be reconstructed as [0, 1, 2, 3, 4]
        without having to traverse the adjacency to reconstruct the contour.
        Any other ordering will not work with this function.
    """
    def create_polygon(tri_generator):
        """ Recontructs the polygon from the group of triangles """
        first_triangle = list(next(tri_generator)[1])
        return first_triangle + [tr[2] for _, tr in tri_generator]
    # sort function which uses the enumeration index of each triangle
    # to lookup at the polygon id and use it for the grouping.
    group_func = lambda tp: polygon_ids[tp[0]]
    grouped_by_polygon_id = itertools.groupby(enumerate(triangles), key=group_func)

    return [create_polygon(group) for _, group in grouped_by_polygon_id]
