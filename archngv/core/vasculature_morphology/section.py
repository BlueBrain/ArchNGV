""" Contains Section class """

from archngv.core.vasculature_morphology.types import emap, pmap


class Section(object):
    """ Section data structure """

    def __init__(self, point_data, edge_data, edge_indices, annotation):

        self.point_data = point_data
        self.edge_data = edge_data
        self.edge_indices = edge_indices

        if len(annotation) > 0:
            self.annotation = annotation
        else:
            self.annotation = None

    @property
    def edges(self):
        """ Get section edges """
        return self.edge_data[:, emap['edges']]

    @property
    def points(self):
        """ Get section points """
        return self.point_data[:, pmap['xyz']]

    @property
    def radii(self):
        """ Get section node radii """
        return self.point_data[:, pmap['r']]

    @property
    def start_point(self):
        """ Start point of the section """
        return self.points[self.edges[0][0]]

    @property
    def end_point(self):
        """ End point of the section """
        return self.points[self.edges[-1][1]]

    @property
    def segment_indices(self):
        """ get node indices for each edge / segment """
        return self.edge_indices

    @property
    def segments(self):
        """
        A segment is a set of two 3D points connected with an edge
        Returns two 1D arrays with the begining and the end of a segment
        """
        edges = self.edges
        points = self.points
        return points[edges[:, 0]], points[edges[:, 1]]
