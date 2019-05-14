from .adjacency import AdjacencyMatrix
from .incidence import IncidenceMatrix
from functools import partial
from scipy.sparse import csr_matrix
import scipy as sp
import numpy as np


class _BaseGraph(object):

    def __init__(self, edges, labels):

        self._E = edges
        if labels is None:
            labels = np.arange(edges.max() + 1, dtype=np.uintp)

        self._L = labels
        self._I = None  # incidence
        self._A = None  # adjacency

        self._edge_look_up_table = {tuple(row): n for n, row in enumerate(edges)}

    @property
    def labels(self):
        return self._L

    @property
    def edges(self):
        """ Edge list"""
        return self._E

    @property
    def n_vertices(self):
        """ Number of vertices"""
        return self.edges.max()

    @property
    def n_edges(self):
        """ Number of edges """
        return self.edges.shape[0]

    def get_edge(self, edge_index):
        """ Returns the edge that correspond to the given index"""
        return self.edges[edge_index]

    def get_edge_index(self, vertex1, vertex2):
        return self._edge_look_up_table[(vertex1, vertex2)]

    @property
    def adjacency_matrix(self):
        if self._A is None:
            self._A = AdjacencyMatrix(self.edges, labels=self.labels)
        return self._A

    def laplacian_matrix(self, normed=False, return_diag=False, use_out_degree=False):
        return sp.sparse.laplacian.csgraph(self.node_adjacency.M, normed=normed, return_diag=return_diag, use_out_degree=use_out_degree)


class DirectedGraph(_BaseGraph):


    def __init__(self, edges, labels=None):
        super(DirectedGraph, self).__init__(edges, labels)

    def incidence_matrix(self):

        if self._I is None:

            N_E = self.n_edges
            N_V = self.n_vertices

            e1 = np.repeat(np.arange(N_E, dtype=np.int), 2)
            e2 = self.edges.flatten()

            values = np.tile((-1, 1), N_E)
            connec = np.column_stack((e2, e1)).T

            I = sp.sparse.csr_matrix((values, connec), shape=(N_V, N_E), dtype=np.int)

            self._I = IncidenceMatrix(I)

        return self._I

'''
def edge_adjacency(self):
    """ AKA Line graph or line adjacency
    Based on the theorem that L(G) = B.T * B - 2I
    """
    if self._E_A is None:

        B = self.incidence._I
        O = sp.sparse.identity(self.n_edges, dtype=np.int)
        A = B.T.dot(B) - 2 * O

        self._E_A = AdjacencyMatrix(A)

    return self._E_A
'''

'''
class UndirectedGraph(_BaseGraph):


    def __init__(self, edges):
        #dA = adj.directed_to_undirected(dA)
        super(UndirectedGraph, self).__init__(edges)

    @property
    def adjacency_matrix(self):
        """ Adjacency Matrix."""
        n_edges = self.edges.shape[0]
        n_points = self.vertices.size
        values = np.ones(2 * n_edges, dtype=np.int)
        connec = np.column_stack((self._E.T, self._E.T[::-1]))
        return sp.sparse.csr_matrix((values, connec), shape=(n_points, n_points), dtype=np.int)

    @property
    def incidence_matrix(self):
        """ Incidence Matrix for undirected graph.
        1 is assigned for both a,b that link a with b
        """
        n_edges = self.edges.shape[0]
        n_points = self._V.size

        e1 = np.repeat(np.arange(n_edges, dtype=np.int), 2)
        e2 = self.edges.flatten()

        values = np.ones(2 * n_edges, dtype=np.int)
        connec = np.column_stack((e2, e1)).T

        return sp.sparse.csr_matrix((values, connec), shape=(n_points, n_edges), dtype=np.int)

    def iter_chain_paths(self):
        from ._utils import iter_chain_paths
        return iter_chain_paths(self)

    """
    def create_subgraph(self, vertices):
        from copy import copy
        return Graph(copy(self.dA), verts=vertices)

    def create_sections_subgraph(self):
        from copy import copy
        sedges = np.column_stack(get_sections(self.predecessors, self.forks, self.terminations))
        verts = np.unique(sedges.ravel()).sort()
        return Graph(copy(self.dA), vertices=verts)
    """
'''
