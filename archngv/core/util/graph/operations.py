import numpy as np
from scipy import sparse, spatial


def add_edges(graph, edges):
    """ Add a set of edges to the graph."""
    dA = graph.adjacency_matrix
    dA[edges[:, 1], edges[:, 0]] = True


def connect_components(graph, points, checkpoint=None, continue_from_checkpoint=True):
    ''' Connects the disconected components of the graph. For each
    Component the sink and the source nodes are extracted and are used
    to calculate their pairwise distance and connecte the closest pairs
    '''
    from scipy.spatial.distance import cdist
    from copy import deepcopy
    def connected_components(dA, condition=None):

        # number of components and component label array
        nc, cc = sparse.csgraph.connected_components(dA, return_labels=True)

        # counts of the frequency of the integer values
        bc = np.bincount(cc)

        ## sort indices according to values from biggest to smallest
        sc = np.argsort(bc)[::-1]

        # mask in case of a given condition to filter results
        mask = np.ones(sc.shape[0], dtype=np.bool) if condition is None else condition(bc[sc])

        # sorted unique componets biggest to smallest
        # components, frequencies, labels
        return sc[mask], bc[sc][mask], cc

    def min_dist_between_groups(points1, points2):

        ds = cdist(points1, points2)
        min_dist = ds.min()
        min_indx1, min_indx2 = np.unravel_index(ds.argmin(), ds.shape)
        return min_dist, min_indx1, min_indx2


    def find_neighbooring_points(points, cmask, max_radius):

        # the radius of the nearest neighborhood sphere
        r2 = max_radius ** 2 * 1e-5

        # we need to exclude the points belonging to the current component
        imask = ~cmask
        rmask = np.zeros_like(imask, dtype=np.bool)

        # centroid of the current component
        centroid = np.mean(points[cmask], axis=0)

        t = cKDTree(points[imask], copy_data=False)

        while not rmask.any():

            r2 *= 10.
            rmask = imask & (distances_from_centroid < r2)

        return rmask

    dA = deepcopy(graph.adjacency_matrix).tolil()

    if checkpoint is not None and continue_from_checkpoint:

        cp_edges = np.load(checkpoint)

        n_cpe_edges = cp_edges.shape[0]

        edges[:n_cpe_edges] = cp_edges

        dA[cp_edges[:, 1], cp_edges[:, 0]] = True

        n = n_cpe_edges
    else:


    components, bc, labels = connected_components(dA)

    ptp = np.linalg.norm(np.ptp(points, axis=0))


    edges = np.zeros((len(components), 2))


        n = 0

    while len(components) > 1:

        #   for i, component in enumerate(components[1::]):

        # get the mask for the points that correspond to the
        I = components[1]

        # current component
        cc_pmask = labels == I
        rm_pmask = ~cc_pmask

        # get the respective indices
        cc_idx = np.where(cc_pmask)[0]
        rm_idx = np.where(rm_pmask)[0]

        # get the respective points
        cc_points = points[cc_pmask]
        rm_points = points[rm_pmask]

        if cc_idx.size < 200:


            _, ind1, ind2 = min_dist_between_groups(cc_points, rm_points)

        else:


            t1 = spatial.cKDTree(cc_points, copy_data=False)
            t2 = spatial.cKDTree(rm_points, copy_data=False)

            r = 1.
            not_found = True
            while not_found:

                res = np.array(t1.query_ball_tree(t2, r))

                nz = np.nonzero(res)[0]

                not_found = nz.size == 0

                r += 5.

            ind1 = nz[0]
            ind2 = res[ind1][0]

        dA[rm_idx[ind2], cc_idx[ind1]] = True

        edges[n, 0] = rm_idx[ind2]
        edges[n, 1] = cc_idx[ind1]

        n += 1


        components, _, labels = connected_components(dA)


    return np.array(edges[:n], dtype=np.intp)


def color_types_via_components(graph, edges):

    _, _, labels = graph.connected_components()

    return labels[edges[:, 0]]
