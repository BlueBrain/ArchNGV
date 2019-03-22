import numpy as _np
import scipy as _sp
from .graphs import DirectedGraph as _DiGraph

_terms_mask = lambda graph: degrees(graph) == 1
_forks_mask = lambda graph: outdegrees(graph) >= 2
_conts_mask = lambda graph: (indegrees(graph) == 1) & (outdegrees(graph) == 1)


def number_of_loops(graph):
    dA = graph.adjacency_matrix
    return dA.diagonal().sum()


def continuations(graph):
    return graph.labels[_conts_mask(graph)]


def terminations(graph):
    return graph.labels[_terms_mask(graph)]


def forks(graph):
    return graph.labels[_forks_mask(graph)]


def outdegrees(graph):
    dA = graph.adjacency_matrix.M
    return dA.sum(axis=0).A[0]


def indegrees(graph):
    dA = graph.adjacency_matrix.M
    return dA.sum(axis=1).A.T[0]


def degrees(graph):
    return indegrees(graph) + outdegrees(graph)


def sources(graph):
    return graph.vertices[indegrees(graph) == 0]


def sinks(graph):
    return graph.vertices[outdegrees(graph) == 0]


def isolated_vertices(graph):
    return graph.vertices[degrees(graph) == 0]


def laplacian(dA, normed=False, return_diag=False, use_out_degree=False):
    return sp.sparse.laplacian(dA, normed=normed, return_diag=return_diag, use_out_degree=use_out_degree)


def chains(graph):
    """
    Returns a list with the chain edges
    """
    from ._utils import get_chains
    return get_chains(graph)


def connected_components(graph, condition=None):

    dA = graph.adjacency_matrix.M

    # number of components and component label array
    nc, cc = _sp.sparse.csgraph.connected_components(dA, return_labels=True)

    # counts of the frequency of the integer values
    bc = _np.bincount(cc)

    # sort indices according to values from biggest to smallest
    sc = _np.argsort(bc)[::-1]

    # mask in case of a given condition to filter results
    mask = _np.ones(sc.shape[0], dtype=_np.bool) if condition is None else condition(bc[sc])

    # sorted unique componets biggest to smallest
    # components, frequencies, labels
    return sc[mask], bc[sc][mask], cc


def number_of_connected_components(graph):
	return _sp.sparse.csgraph.connected_components(graph.dA, directed=isinstance(graph, DirectedGraph), return_labels=False)


def traversal(graph, start_node=0, method='dfs', return_predecessors=False, condition=None):
    _SEARCH_ORDER = {'dfs': _sp.sparse.csgraph.depth_first_order,
                     'bfs': _sp.sparse.csgraph.breadth_first_order}

    dA = graph.adjacency_matrix.T
    is_directed = isinstance(graph, _DiGraph)

    if return_predecessors:

        vertices, predecessors = _SEARCH_ORDER[method](dA, start_node, return_predecessors=return_predecessors,
                                 directed=is_directed)

        predecessors[predecessors == -9999] = -1

        return vertices, predecessors

    else:

        return _SEARCH_ORDER[method](dA, start_node, return_predecessors=return_predecessors,
                                 directed=is_directed)
