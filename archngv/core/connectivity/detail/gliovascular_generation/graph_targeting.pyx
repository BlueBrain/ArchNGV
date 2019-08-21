
import numpy as np
cimport numpy as np

from libc.math cimport round


cpdef create_targets(points, edges, parameters):
    """ Distributes points across the edges of the graph without taking into
    account the geometrical characteristics of the data structure such as
    vasculature thickness.
    """
    seg_begs = points[edges[:, 0]].astype(np.float64)
    seg_ends = points[edges[:, 1]].astype(np.float64)

    target_points, edges_idx = distribution_on_line_graph(seg_begs,
                                                          seg_ends,
                                                          float(parameters['linear_density']))

    return target_points, edges_idx


cpdef distribution_on_line_graph(double[:, :] segment_starts, double[:, :] segment_ends, double linear_density):
    """
    Distributes points with respect to linear density on the
    linea graph, the connectivity of which is specified through
    adjacency matrix dA and with nodes that correspond to points
    """
    cdef :

        double[:, :] targets, seg_vecs
        double[:] seg_lens
        unsigned long[:] seg_idx

        double u_vec_x, u_vec_y, u_vec_z
        double s_vec_x, s_vec_y, s_vec_z

        double cum_len, v_len, dy

        unsigned long  N_targets, s_index = 0, n = 0

    dy = 1. / linear_density

    seg_vecs = np.subtract(segment_ends, segment_starts)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)

    N_targets = <unsigned long>round(np.sum(seg_lens) * linear_density)

    # initialize result array
    targets = np.zeros((N_targets, 3), dtype=np.float64)
    seg_idx = np.zeros(N_targets, dtype=np.uintp)

    cum_len = - dy

    while n < N_targets:

        v_len = seg_lens[s_index]

        if v_len > 0.:

            u_vec_x = seg_vecs[s_index, 0] / v_len
            u_vec_y = seg_vecs[s_index, 1] / v_len
            u_vec_z = seg_vecs[s_index, 2] / v_len

            while  cum_len + dy <= v_len and n < N_targets:

                cum_len = cum_len + dy

                seg_idx[n] = s_index

                targets[n, 0] = segment_starts[s_index, 0] + u_vec_x * cum_len
                targets[n, 1] = segment_starts[s_index, 1] + u_vec_y * cum_len
                targets[n, 2] = segment_starts[s_index, 2] + u_vec_z * cum_len

                n = n + 1

            cum_len = cum_len - v_len

        s_index = s_index + 1

    return np.asarray(targets[:n]), np.asarray(seg_idx[:n])
