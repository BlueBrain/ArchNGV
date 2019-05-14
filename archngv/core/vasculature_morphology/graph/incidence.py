""" Incidence Matrix """

class IncidenceMatrix(object):
    """ Incidence matrix constructed from scipy csgraph
    """
    def __init__(self, csgraph):

        self.M = csgraph

    def find_edges_with_vertex(self, vertex_index):
        """ Returns all the edges that contain the given vertex
        """
        return self.M.getrow(vertex_index).nonzero()[1]

    def find_vertices_with_edge(self, edge_index):
        """ Returns all vertices that are contained in edge
        """
        return self.M.getcol(edge_index).nonzero()[0]
