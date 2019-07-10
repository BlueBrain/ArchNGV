""" Remap """

import numpy as np


def remap_edge_vertices(edge_data):
    """ Remap edge indices to account
    for removed edges and indices that are not
    used after the removal.
    """
    new_data = edge_data.copy()

    edges = new_data[:, 0:3]

    e = edges.ravel()

    _, b, c = np.unique(e, return_index=True, return_inverse=True)

    rmp = np.arange(b.size, dtype=np.uintp)

    e[:] = rmp[c]

    return new_data
