from scipy import spatial


def nearest_neighbor_distances(masked_points, all_points):
    t = spatial.cKDTree(all_points)

    # ignore distance to self by calc the two closest neighbors and
    # picking the second one
    return t.query(masked_points, 2)[0][:, 1]
