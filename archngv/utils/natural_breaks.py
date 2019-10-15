""" Natural breaks algorithms
"""
import numpy as np


def _ssq(j, i, sums, sums_of_squares):
    """
    sji = s(j, i)
    """
    if j > 0:
        muji = (sums[i] - sums[j - 1]) / (i - j + 1)
        sji = sums_of_squares[i] - sums_of_squares[j - 1] - (i - j + 1) * muji ** 2
    else:
        sji = sums_of_squares[i] - sums[i] ** 2 / (i + 1)

    return 0. if sji < 0. else sji


def _fill_row_k(imin, imax, cluster, S, backtracking_matrix, sums, sums_of_squares, n_values):
    """
    Function that recursively divides and conquers computations for cluster j

    Args:
        imin: int
            Minimum index in cluster to be computed
        imax: int
            Maximum index in cluster to be computed
    """
    if imin > imax:
        return

    # start at midpoint between imin and imax
    i = (imin + imax) // 2
    S[cluster][i] = S[cluster - 1][i - 1]
    backtracking_matrix[cluster, i] = i

    # lower end for j
    jlow = cluster

    if imin > cluster:
        jlow = max(jlow, backtracking_matrix[cluster][imin - 1])
    jlow = max(jlow, backtracking_matrix[cluster - 1, i])

    # the upper end for j
    jhigh = i - 1
    if imax < n_values - 1:
        jhigh = min(jhigh, backtracking_matrix[cluster][imax + 1])

    for j in range(jhigh, jlow - 1, -1):

        sji = _ssq(j, i, sums, sums_of_squares)

        if sji + S[cluster - 1, jlow - 1] >= S[cluster][i]:
            break

        # Examine the lower bound of the cluster border, compute s(jlow, i)
        sjlowi = _ssq(jlow, i, sums, sums_of_squares)

        SSQ_jlow = sjlowi + S[cluster - 1][jlow - 1]

        if SSQ_jlow < S[cluster][i]:
            # shrink the lower bound
            S[cluster, i] = SSQ_jlow
            backtracking_matrix[cluster][i] = jlow

        jlow += 1

        SSQ_j = sji + S[cluster - 1][j - 1]
        if SSQ_j < S[cluster][i]:
            S[cluster, i] = SSQ_j
            backtracking_matrix[cluster, i] = j

    _fill_row_k(imin, i - 1, cluster, S, backtracking_matrix, sums, sums_of_squares, n_values)
    _fill_row_k(i + 1, imax, cluster, S, backtracking_matrix, sums, sums_of_squares, n_values)


def _fill_dp_matrix(data, S, backtracking_matrix, n_clusters, n_values):
    """
    This is a dynamic programming way to solve the problem of minimizing
    within-cluster sum of squares. It's similar to linear regression
    in this way, and this calculation incrementally computes the
    sum of squares that are later read.
    """
    sums = np.zeros(n_values, dtype=np.float64)
    sums_of_squares = np.zeros(n_values, dtype=np.float64)

    # median. used to shift the values of x to improve numerical stability
    shift = data[n_values // 2]

    for i in range(n_values):
        if i == 0:
            sums[0] = data[0] - shift
            sums_of_squares[0] = (data[0] - shift) ** 2
        else:
            sums[i] = sums[i - 1] + data[i] - shift
            sums_of_squares[i] = sums_of_squares[i - 1] + (data[i] - shift) ** 2

        S[0][i] = _ssq(0, i, sums, sums_of_squares)
        backtracking_matrix[0][i] = 0

    for cluster_index in range(1, n_clusters):
        imin = cluster_index if cluster_index < n_clusters - 1 else n_values - 1
        _fill_row_k(imin, n_values - 1, cluster_index, S, backtracking_matrix, sums, sums_of_squares, n_values)


def kmeans_1d(data, n_clusters):
    """
    Finds the gaps in the ordered data and returns the index for the smallest value
    in each cluster.

    Args:
        data: array[float, (N,)]
            1D array sorted with increasing ordering.
        n_clusters: int
            Number of clusters.

    Returns:
        ids: array[int, (n_clusters,)]

    Example:
        data = np.array([1.1,   1.2,   1.3,  10.1,  10.2,  10.3, 100.1, 100.2, 100.3])
        ids = kmeans_1d(data, 3)
        which should give [0, 3, 6]

    Notes:
        Article doi: 10.32614/RJ-2011-01
        Based on the following implementations:
            https://github.com/simple-statistics/simple-statistics/blob/master/src/ckmeans.js
            https://github.com/llimllib/ckmeans/blob/master/ckmeans.py
    """
    if n_clusters == 1:
        return np.array([0], dtype=np.int)
    if n_clusters == len(data):
        return np.arange(len(data), dtype=np.int)

    n_values = len(data)

    S = np.zeros((n_clusters, n_values), dtype=np.float64)

    backtracking_matrix = np.zeros((n_clusters, n_values), dtype=np.int)

    _fill_dp_matrix(data, S, backtracking_matrix, n_clusters, n_values)

    cluster_right = n_values - 1
    ids = np.empty(n_clusters, dtype=np.int)

    # Backtrack the clusters from the dynamic programming matrix. This
    # starts at the bottom-right corner of the matrix (if the top-left is 0, 0),
    #  and moves the cluster target with the loop.
    for i, cluster in enumerate(range(n_clusters - 1, -1, -1)):

        # the backtrack matrix stores where the cluster should start and end.
        cluster_left = backtracking_matrix[cluster, cluster_right]

        # store the smallest value id in each cluster
        # because we traverse the matric from the end to the start we
        # need to store in reverse the positions in ids
        ids[n_clusters - i - 1] = cluster_left
        if cluster > 0:
            cluster_right = cluster_left - 1

    return ids
