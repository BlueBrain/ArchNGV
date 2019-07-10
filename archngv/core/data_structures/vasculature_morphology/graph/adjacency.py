""" Adjacency matric for graph operations """

import numpy as np
from scipy import sparse


def _create_sparse_matrix(edges, n_vertices, weights):
    """ Create a sparse matric from and edge list """
    return sparse.csr_matrix((weights, edges.T), shape=(n_vertices, n_vertices))


def _return_index(mask, enabled=True):
    """ Return indices where mask is True """
    return np.where(mask)[0] if enabled else mask


class AdjacencyMatrix(object):
    """ Adjacency Matrix using a double sparse representation,
    one for fast column and one for fast row queries.
    """

    def __init__(self, edges, labels=None, weights=None):

        if weights is None:
            weights = np.ones(edges.shape[0], dtype=np.int)

        self._labels = labels

        n_vertices = edges.max() + 1 if labels is None else labels.size

        self.M = _create_sparse_matrix(edges, n_vertices, weights)

        self._idx = self.M.indices

        # indpointer from csr sparse matrix
        self._idp = self.M.indptr

        # transposed csr matrix
        csgraph_T = self.M.T.tocsr()

        # indices from transposed csr sparse matrix
        self._idx_T = csgraph_T.indices

        # indpointer from transposed csr sparse matrix
        self._idp_T = csgraph_T.indptr

    @property
    def labels(self):
        """ Vertex labels if any """
        return self._labels

    @property
    def number_of_self_loops(self):
        """ Vertices that loop to themselves """
        return np.count_nonzero(self.M.diagonal())

    @property
    def outdegrees(self):
        """
        Summing the adjancency matrix over the columns returns the number of edges that come out
        from each vertec.
        """
        return (self.M != 0).sum(axis=1).A.T[0]

    @property
    def indegrees(self):
        """
        Summing the adjacency matrix over the rows returns the number of edges that come in each
        vertex.
        """
        return (self.M != 0).sum(axis=0).A[0]

    @property
    def degrees(self):
        """ The degree of each vertex is the sum of all the incoming and outcoming edges.
        """
        return self.indegrees + self.outdegrees

    def forks(self, return_index=True):
        """ Forking vertices """
        mask = self.outdegrees >= 2
        return _return_index(mask, enabled=return_index)

    def sources(self, return_index=True):
        """ Source vertices """
        mask = self.indegrees == 0
        return _return_index(mask, enabled=return_index)

    def sinks(self, return_index=True):
        """ Sink vertices """
        mask = self.outdegrees == 0
        return _return_index(mask, enabled=return_index)

    def terminations(self, return_index=True):
        """ Termination vertices """
        mask = self.degrees == 1
        return _return_index(mask, enabled=return_index)

    def continuations(self, return_index=True):
        """ Vertices that have one parent and one child """
        mask = (self.indegrees == 1) & (self.outdegrees == 1)
        return _return_index(mask, enabled=return_index)

    def isolated_vertices(self, return_index=True):
        """ Unconnected vertices """
        mask = self.degrees == 0
        return _return_index(mask, enabled=return_index)

    def children(self, node):
        """ Get the parents of a given vertex."""
        return self._idx[self._idp[node]: self._idp[node + 1]]

    def parents(self, node):
        """ Get the children of a given vertex."""
        return self._idx_T[self._idp_T[node]: self._idp_T[node + 1]]

    def neighbors(self, vertex_index):
        """ Get all adjacent vertices to the given vertex."""

        vertices = set()

        for p in self.parents(vertex_index):
            vertices.add(p)
        for c in self.children(vertex_index):
            vertices.add(c)

        return sorted(list(vertices))

    def connected_components(self, condition=None):
        """ Returns connected components in the matrix """
        # number of components and component label array
        _, cc = sparse.csgraph.connected_components(self.M, return_labels=True)

        # counts of the frequency of the integer values
        bc = np.bincount(cc)

        # sort indices according to values from biggest to smallest
        sc = np.argsort(bc)[::-1]

        # mask in case of a given condition to filter results
        mask = np.ones(sc.shape[0], dtype=np.bool) if condition is None else condition(bc[sc])

        # sorted unique componets biggest to smallest
        # components, frequencies, labels
        return sc[mask], bc[sc][mask], cc

    def fast_traversal(self, start_node=0, method='dfs', return_predecessors=False, directed=False):
        """ Perform fast traversal """
        _SEARCH_ORDER = {'dfs': sparse.csgraph.depth_first_order,
                         'bfs': sparse.csgraph.breadth_first_order}

        dA = self.M

        if return_predecessors:

            vertices, predecessors = _SEARCH_ORDER[method](dA, start_node, return_predecessors=return_predecessors,
                                     directed=directed)

            predecessors[predecessors == -9999] = -1

            return vertices, predecessors

        else:

            return _SEARCH_ORDER[method](dA, start_node, return_predecessors=return_predecessors,
                                     directed=directed)


def directed_to_undirected(dA):
    """
    Convert a directed adjacency boolean matrix to an undirected one. The transpose is added
    to the matrix and the lower triangular values are returned.
    """
    return sparse.tril(dA + dA.T, format=dA.format)
