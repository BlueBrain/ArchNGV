""" Operation functions that return a new vasculature object with the modified data """

import numpy as np
from .vasculature import Vasculature
from .io.convert_to_spec import convert_to_spec
from ..util.hexagonal_masking import mask_indices_by_geometry


def mask_by_geometry(V, geometry):
    """ Filter out any edges that are not included in the geometry """

    E = V.edges

    # indices of point array
    N = np.arange(V.data.points.shape[0])

    # indices that remain
    R = mask_indices_by_geometry(V.points, geometry)

    # indices that will be removed
    D = np.setdiff1d(N, R)

    mask = ~(np.in1d(E[:, 0], D) | np.in1d(E[:, 1], D))

    dA = V.point_graph.adjacency_matrix.M.copy()

    new_edges = np.column_stack(dA[R, :][:, R].nonzero())

    new_points, new_radii = V.points[R], V.radii[R]

    new_types = V.segment_types[mask]

    return Vasculature(*convert_to_spec(new_points, new_edges, new_radii, new_types))


def mask_by_bounding_box(V, bb, scale=1.):
    """ Filter out any edges that are not included in the bounding box """

    E = V.edges

    # indices of point array
    N = np.arange(V.points.shape[0])

    # indices that remain
    p = V.points

    mask = bb.points_inside(p)

    """
    mask = (p[:, 0] >= (2. - scale) * bb[0][0]) & (p[:, 0] < scale * bb[1][0]) & \
           (p[:, 1] >= (2. - scale) * bb[0][1]) & (p[:, 1] < scale * bb[1][1]) & \
           (p[:, 2] >= (2. - scale) * bb[0][2]) & (p[:, 2] < scale * bb[1][2])
    """

    R = np.where(mask)[0]

    D = np.setdiff1d(N, R)
    mask = ~(np.in1d(E[:, 0], D) | np.in1d(E[:, 1], D))

    dA = V.point_graph.adjacency_matrix.M.copy()

    new_edges = np.column_stack(dA[R, :][:, R].nonzero())

    new_points, new_radii = V.points[R], V.radii[R]

    new_types = V.segments_types[mask]

    return Vasculature(*(convert_to_spec(new_points, new_edges, new_radii, new_types)))
