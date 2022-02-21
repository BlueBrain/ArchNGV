"""Binning utils"""
import math

import numpy as np


def rebin_counts(ref_counts, ref_bin_edges, bin_edges):
    """Redistributes the ref_counts from the ref_bin_edges to bin_edges"""
    if len(ref_bin_edges) < 2 or len(bin_edges) < 2:
        raise ValueError("At least two bins required.")

    le = lambda v1, v2: v1 < v2 or np.isclose(v1, v2, rtol=1e-6)
    ge = lambda v1, v2: v1 > v2 or np.isclose(v1, v2, rtol=1e-6)

    n_bins, n_ref_bins = len(bin_edges) - 1, len(ref_bin_edges) - 1

    new_counts = np.zeros(n_bins, dtype=ref_counts.dtype)

    i = j = 0
    total = 0.0
    while i < n_bins and j < n_ref_bins:

        i_left, i_right = bin_edges[i], bin_edges[i + 1]
        j_left, j_right = ref_bin_edges[j], ref_bin_edges[j + 1]
        counts = ref_counts[j]

        # i: | |
        # j:   | |
        if le(i_right, j_left):
            i += 1
        # i:   | |
        # j: | |
        elif le(j_right, i_left):
            j += 1
        # i:  | |
        # j: |   |
        elif ge(i_left, j_left) and le(i_right, j_right):
            total += counts * (i_right - i_left) / (j_right - j_left)
            new_counts[i] = total
            total = 0.0
            i += 1
        # i: |   |
        # j:  | |
        elif le(i_left, j_left) and ge(i_right, j_right):
            total += counts
            j += 1
            if math.isclose(i_right, j_right):
                new_counts[i] = total
                total = 0.0
                i += 1
        # i: | |
        # j:  | |
        elif i_left < j_left < i_right:
            total += counts * (i_right - j_left) / (j_right - j_left)
            new_counts[i] = total
            total = 0.0
            i += 1
        # i:  | |
        # j: | |
        elif i_left < j_right < i_right:
            total += counts * (j_right - i_left) / (j_right - j_left)
            j += 1
        else:
            raise ValueError(f"{i_left},  {i_right}, {j_left}, {j_right}")

        if i == n_bins - 1:
            new_counts[i] = total

    return new_counts
