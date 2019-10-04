""" Graphs """

import numpy as np
import scipy as sp

from archngv.core.vasculature_morphology.graph.adjacency import AdjacencyMatrix
from archngv.core.vasculature_morphology.graph.incidence import IncidenceMatrix


class BaseGraph(object):
    """ Basic graph data structure
    """
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
        """ Vertex labels """
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
        """ Find edge index that connects the two vertices """
        return self._edge_look_up_table[(vertex1, vertex2)]

    @property
    def adjacency_matrix(self):
        """ Return adjacency matrix """
        if self._A is None:
            self._A = AdjacencyMatrix(self.edges, labels=self.labels)
        return self._A

    def laplacian_matrix(self, normed=False, return_diag=False, use_out_degree=False):
        """ Return the laplacian matrix """
        return sp.sparse.csgraph.laplacian(self.adjacency_matrix.M,
                                          normed=normed,
                                          return_diag=return_diag,
                                          use_out_degree=use_out_degree)


class DirectedGraph(BaseGraph):
    """ Directed graph
    """

    def __init__(self, edges, labels=None):
        super(DirectedGraph, self).__init__(edges, labels)

    def incidence_matrix(self):
        """ Return adjacency matrix """
        if self._I is None:

            N_E = self.n_edges
            N_V = self.n_vertices

            e1 = np.repeat(np.arange(N_E, dtype=np.int), 2)
            e2 = self.edges.flatten()

            values = np.tile((-1, 1), N_E)
            connec = np.column_stack((e2, e1)).T

            _I = sp.sparse.csr_matrix((values, connec), shape=(N_V, N_E), dtype=np.int)

            self._I = IncidenceMatrix(_I)

        return self._I
