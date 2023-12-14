"""
Functions related to triangles
"""
import math
from typing import List

import numpy as np

from archngv.utils.linear_algebra import normalize_vectors, rowwise_dot


def vectorized_consecutive_triangle_vectors(points, triangles):
    """Sequential face vectors"""
    return (
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )


def vectorized_triangle_normal(vectors1, vectors2):
    """
    Args:
        vectors1: 2D array
            Vectors where each row represents one side of a triangle.
        vectors2: 2D array
            Vectors where each row represents second side a triangle.

    Returns:
        2D array:
            Normal vectors of the triangle faces.
    """
    crosses = np.cross(vectors1, vectors2)
    return normalize_vectors(crosses)


def vectorized_parallelogram_area(vectors1, vectors2):
    """
    Args:
        vectors1: 2D array
            Vectors where each row represents one side of a parallelogram.
        vectors2: 2D array
            Vectors where each row represents second side a paralellogram.
        The vectors must not be parallel.

    Returns:
        1D array:
            Areas of the parallelograms defined by vectors1 and vectors2
    """
    return np.sqrt(np.sum(np.cross(vectors1, vectors2) ** 2, axis=1))


def vectorized_triangle_area(vectors1, vectors2):
    """
    Args:
        vectors1: 2D array
            Vectors where each row represents one side of a parallelogram.
        vectors2: 2D array
            Vectors where each row represents second side a paralellogram.

    Returns:
        1D array:
            Areas of the triangles defined by vectors1 and vectors2
    """
    return vectorized_parallelogram_area(vectors1, vectors2) * 0.5


def vectorized_parallelepiped_volume(vectors1, vectors2, vectors3):
    """
    Args:
        vectors1: 2D array
            Vectors where each row represents one side of a parallelepiped.
        vectors2: 2D array
            Vectors where each row represents second side a parallelepiped.
        vectors3: 2D array
            Vectors where each row represents third side a parallelepiped.

    Returns:
        1D array:
            Volumes of the parallelepipeds defined by vectors1, vectors2 and vectors3
    """
    return np.abs(rowwise_dot(np.cross(vectors1, vectors2), vectors3))


def vectorized_tetrahedron_volume(vectors1, vectors2, vectors3):
    """
    Args:
        vectors1: 2D array
            Vectors where each row represents one side of a tetrahedron.
        vectors2: 2D array
            Vectors where each row represents second side a tetrahedron.
        vectors3: 2D array
            Vectors where each row represents third side a tetrahedron.

    Returns:
        1D array:
            Volumes of the tetrahedrons defined by vectors1, vectors2 and vectors3
    """
    return (1.0 / 6.0) * vectorized_parallelepiped_volume(vectors1, vectors2, vectors3)


def subdivide_triangles_by_total_area(points, triangles, min_number_of_points):
    """Splits each triangle into three triangles with a new point at
    the center of the parent triangle. The total area of all triangles
    is used in order to determine the distribution of the number of
    points depending on the area of each triangle. Big triangles will
    have more points that small ones, according to the point area density
    calculated from the min_area_of_points.

    Given a number of points per triangle, the number of splitings has
    to be determined in order to achieve at least that number of points.
    N_level = 3 ^ L where N_level is the number of points on that level,
    and L the subsequent splitings.

    Thus, the series S[0, l](3^l) = 0.5(3^(l + 1) - 1) = N_total, where
    N_total is the total number of points after l splitings. Solving for
    the level:

    level = log3(2 * N_total + 1) - 1

    That is the maximum level that we reach by subsequent splitings in order
    to create N_total points.

    Args:
        points: array[float, (N, 3)]
        triangles: array[int, (M, 3)]
        target_number_of_points: int
            The minimum number of points to generate on the faces
            of the triangles.

    Returns:
        points: array[float, (N + K, 3)
            The initial array plus the added center points
        triangles: array[int, (W, 3)]
            A new set of subdivided triangles
    """
    points = np.asarray(points)
    triangles = np.asarray(triangles)

    if len(points) >= min_number_of_points:
        return points, triangles

    areas = vectorized_triangle_area(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )

    # don't take into account the existing points
    density = float(min_number_of_points - len(points)) / areas.sum()

    result_points = list(points)
    result_triangles = []

    sorted_ids = np.argsort(areas, kind="stable")[::-1]
    sorted_triangles = triangles[sorted_ids].tolist()
    points_per_triangle = np.rint(areas[sorted_ids] * density)

    for i, triangle in enumerate(sorted_triangles):
        n_points = points_per_triangle[i]

        try:
            max_level = int(np.ceil(math.log(2.0 * n_points + 1.0, 3)) - 1.0)
        except ValueError:
            continue

        new_points, new_triangles = subdivide_triangles(
            result_points, [triangle], max_level=max_level, max_points=n_points
        )

        result_points.extend(new_points)
        result_triangles.extend(new_triangles)

        if len(result_points) >= min_number_of_points:
            result_triangles.extend(sorted_triangles[i + 1 : :])
            break

    return np.asarray(result_points, dtype=points.dtype), np.asarray(
        result_triangles, dtype=triangles.dtype
    )


def subdivide_triangles(initial_points, initial_triangles, max_level=0, max_points=np.inf):
    """Add face centers of triangles to the available points for the orientations.
    At each iteration all current triangles are substituted by their respective
    split triplets.

    Returns:
        added_points: array[float, (K, 3)]
            The new points that are created from the subdivisions.
        new_triangles: array[int, (M, 3)]
            The new split triangles from the subdivisions.
    """
    points = list(initial_points)  # copy
    new_triangles = list(initial_triangles)  # copy

    added_points = []

    offset = len(points)

    level = 0
    while level <= max_level and len(added_points) <= max_points:
        iteration_points = []
        iteration_triangles = []

        for i, triangle in enumerate(new_triangles):
            center = (points[triangle[0]] + points[triangle[1]] + points[triangle[2]]) / 3.0

            iteration_points.append(center)
            iteration_triangles.extend(
                [
                    (triangle[0], offset, triangle[1]),
                    (triangle[1], offset, triangle[2]),
                    (triangle[2], offset, triangle[0]),
                ]
            )

            offset += 1

            if len(added_points) + i + 1 >= max_points:
                break

        # the big triangles are substituted by
        # their children triplets
        new_triangles = iteration_triangles

        points.extend(iteration_points)
        added_points.extend(iteration_points)

        level += 1

    return added_points, new_triangles


def globally_ordered_verts(face_points, face_vertices):
    """Given a chain of face_vertices, find how to traverse it
    via face_point coordinate ordering in order to achieve the same
    ordering from any overlapping polygon

    Args:
        face_points: array[float, (N, 3)]
        face_vertices: array[int, (M,)]

    Returns:
        face_vertices_ordering: array[ing, (M, )]
    """
    # the start vertex of the face is selected be the
    # ordering of the face coordinates by zyx
    first_index, second_index = np.lexsort(face_points.T)[:2]

    n_verts = len(face_points)

    if first_index > second_index or (first_index == 0 and second_index == n_verts - 1):
        # if the second index is smaller than the first it means that we need to
        # iterate the vertices backwards starting from the second_index
        # By shifting the array by len(face_vertices) - second_index - 1 we make
        # sure that the second_index is at the end of the array eg
        # [4, 5, 6, 7, 8] -> [7, 8, 4, 5, 6] if first_index = 2, second_index = 1
        # Then we reverse the array to [6, 5, 4, 8, 7] to achieve the same ordering
        right_shift = len(face_vertices) - first_index - 1
        return np.roll(face_vertices, right_shift)[::-1]

    # shift the array to the left so that first_index element is first eg
    # [5, 6, 7, 8] -> [7, 8, 5, 6] if first_index = 2
    return np.roll(face_vertices, -first_index)


def polygons_to_triangles(points, face_vertices_collection):
    """Triangles from polygons

    Args:
        points: array[float, (N, 3)]
            The 3D coordinates of the polygons
        face_vertices_collection: list[list[int]]
            A list of polygons the vertices in consecutive order

    Returns:
        triangles: array[uint, (M, 3)]
        triangle_to_polygon_map: array[uint, (M,)]

    Notes:
        This algorithm works by splitting a polygon into consecutive
        triangles, starting from an existing vertex and traversing
        the polygon clockwise or counterclockwise depending on the global
        order determined by the coordinates of the polygon.

        A triangle consists for three vertices and for any extra vertex
        we have in the polygon creates a new triangle, i.e.:

        3 vertices 1 triangle
        4 vertices 2 triangles
        5 vertices 3 triagnels
        n vertices n - 2 triangles

    """
    n_tris = sum(len(verts) - 2 for verts in face_vertices_collection)

    tris = np.empty((n_tris, 3), dtype=np.uint64)

    # maps triangles back to the polygon list
    tris_to_polys_map = np.empty(n_tris, dtype=np.uint64)

    n = 0
    for face_index, face_vertices in enumerate(face_vertices_collection):
        n_vertices = len(face_vertices)
        face_vertices = np.asarray(face_vertices, dtype=np.uintp)

        # triangle, store it as it is
        if n_vertices == 3:
            tris[n] = face_vertices
            tris_to_polys_map[n] = face_index
            n += 1
            continue

        face_points = points[face_vertices]

        # consecutive vertices ordered by a global ordering of their
        # coordinates.
        o_verts = globally_ordered_verts(face_points, face_vertices)

        # split the polygon into consecutive triangles
        for i in range(2, n_vertices):
            # i = 2 -> 0, 1, 2
            # i = 3 -> 0, 2, 3 etc.
            tris[n] = o_verts[0], o_verts[i - 1], o_verts[i]

            # keep track of the polygon we started from
            tris_to_polys_map[n] = face_index
            n += 1

    return tris, tris_to_polys_map


def local_to_global_triangles(triangles, ps_tris_offsets, local_to_global_vertices):
    """Converts and array from the local index space to the global one

    Args:
        triangles: array[int, (N, 3)]
        ps_tris_offsets: array[int, (N + 1,)]
        local_to_global_vertices: array[int, (M,)]
    """
    from archngv.utils.functional import consecutive_pairs

    global_tris = np.empty_like(triangles)
    total_tris = 0
    for (p_beg, t_beg), (p_end, t_end) in consecutive_pairs(ps_tris_offsets):
        # get the local triangles for the i-th astrocyte
        local_tris = triangles[t_beg:t_end]
        n_tris = len(local_tris)

        # get the local_to_global vertex map slice for the i-th asotrycte
        l2g = local_to_global_vertices[p_beg:p_end]

        # using the local to global map to transform the local triangle vertices
        # to the global unique ones
        global_tris[total_tris : total_tris + n_tris] = l2g[local_tris]
        total_tris += n_tris

    return global_tris


def local_to_global_mapping(points, triangles, ps_tris_offsets, triangle_labels=None, decimals=4):
    """Given an array of points return an array of indices that correspond
    to all the unique points in the array.

    1D Example:

    arr = [0.22, 0.11, 0.22, 0.33, 0.11, 0.44, 0.44, 0.44, 0.44, 0.0]
    vertices = [2, 1, 2, 3, 1, 4, 4, 4, 4, 0]

    """
    from archngv.utils.geometry import unique_points

    unique_idx, ps_to_uverts_map = unique_points(points, decimals=decimals)
    global_tris = local_to_global_triangles(triangles, ps_tris_offsets, ps_to_uverts_map)

    # because vertices array has the same unique vertex id for duplicate coordinates
    # when we remapped the triangles we actually mapped to the unique index space.
    # Finally we remove the duplicate triangles via unique across rows after we make
    # sure that all the triangle ids are sorted
    sorted_cols_tris = np.sort(global_tris, axis=1, kind="mergesort")
    _, idx = np.unique(sorted_cols_tris, axis=0, return_index=True)

    # keep the initial order of the triangles
    # when selecting unique rows
    idx.sort(kind="mergesort")

    global_tris = global_tris[idx]

    if triangle_labels is None:
        return points[unique_idx], global_tris
    return points[unique_idx], global_tris, triangle_labels[idx]


def local_to_global_polygon_ids(polygon_ids):
    """Given an array of increasing polygon_ids stored in the local index space
    Example:

        polygon_ids = [0, 0, 1, 1, 0, 1, 1, 2]
        global_poly = [0, 0, 1, 1, 2, 3, 3, 4]
    """
    is_different = np.empty(polygon_ids.shape, dtype=bool)
    is_different[0] = False
    # check if the i and i-1 ids are different
    is_different[1:] = polygon_ids[1:] != polygon_ids[:-1]
    # the cumulative sum trackes the previous values so
    # that the index always increases with a new non duplicate value
    return np.cumsum(is_different)


def triangles_to_polygons(triangles: np.ndarray, polygon_ids: np.ndarray) -> List[List[int]]:
    """Converts triangles to a polygon list

    Args:
        triangles:
            Integer array of triangles (N, 3) in specific order to reconstruct polygons
        polygon_ids:
            Integer array of the polygon id (N,) that each triangle belongs to.
    Returns:
        List of polygons, where each polygon is a list of unique vertices.

    Notes:
        Triangles have to be ordered in the following way:
            [0, 1, 2],
            [0, 2, 3],
            [0, 2, 4]
        so that the polygon can right away be reconstructed as [0, 1, 2, 3, 4]
        without having to traverse the adjacency to reconstruct the contour.
        Any other ordering will not work with this function.
    """
    from itertools import groupby

    def create_polygon(tri_generator):
        """Reconstructs the polygon from the group of triangles"""
        first_triangle = list(next(tri_generator)[1])
        return first_triangle + [tr[2] for _, tr in tri_generator]

    # sort function which uses the enumeration index of each triangle
    # to lookup at the polygon id and use it for the grouping.
    # groupby keys must be sorted to work
    group_func = lambda tp: polygon_ids[tp[0]]
    grouped_by_polygon_id = groupby(enumerate(triangles), key=group_func)

    return [create_polygon(group) for _, group in grouped_by_polygon_id]
