"""graph targetting"""
import numpy as np


def create_targets(points, edges, parameters):
    """Distributes points across the edges of the graph without taking into
    account the geometrical characteristics of the data structure such as
    vasculature thickness.
    """
    seg_begs = points[edges[:, 0]].astype(np.float64)
    seg_ends = points[edges[:, 1]].astype(np.float64)

    target_points, edges_idx = _distribution_on_line_graph(
        seg_begs, seg_ends, float(parameters["linear_density"])
    )

    return target_points, edges_idx


def _distribution_on_line_graph(segment_starts, segment_ends, linear_density):
    """Distributes points with respect to linear density"""

    seg_vecs = np.subtract(segment_ends, segment_starts)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)

    N_targets = int(round(np.sum(seg_lens) * linear_density))

    targets = np.empty((N_targets, 3), dtype=np.float64)
    seg_idx = np.empty(N_targets, dtype=np.uintp)

    dy = 1.0 / linear_density
    cum_len = -dy

    s_index = n = 0
    while n < N_targets:
        v_len = seg_lens[s_index]

        if v_len <= 0.0:
            s_index += 1
            continue

        u_vec = seg_vecs[s_index, :] / v_len

        while cum_len + dy <= v_len and n < N_targets:
            cum_len += dy
            seg_idx[n] = s_index
            targets[n, :] = segment_starts[s_index, :] + u_vec * cum_len
            n += 1

        cum_len -= v_len
        s_index += 1

    return targets[:n], seg_idx[:n]
