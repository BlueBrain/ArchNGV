import numpy as np
from copy import deepcopy

def remap_edge_vertices(edge_data):

    new_data = edge_data.copy()

    edges = new_data[:, 0:3]

    e = edges.ravel()

    _, b, c = np.unique(e, return_index=True, return_inverse=True)

    rmp = np.arange(b.size, dtype=np.uintp)

    e[:] = rmp[c]

    return new_data
