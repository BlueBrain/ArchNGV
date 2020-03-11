""" VasculatureAPI wrapper for ArchNGV """
from vasculatureapi import SectionVasculature


class Vasculature:
    """ Vasculature wrapper using VasculatureAPI
    """
    def __init__(self, vasculature):
        self._impl = vasculature

    @classmethod
    def load(cls, filepath):
        """ Load vasculature file """
        section_vasculature = SectionVasculature.load(filepath)
        point_vasculature = section_vasculature.as_point_graph()
        return cls(point_vasculature)

    @property
    def node_properties(self):
        """ Node properties dataframe """
        return self._impl.node_properties

    @property
    def edge_properties(self):
        """ Edge properties dataframe """
        return self._impl.edge_properties

    @property
    def points(self):
        """ Return vasculature points """
        return self._impl.points

    @property
    def edges(self):
        """ Return vasculature edges """
        return self._impl.edges

    @property
    def radii(self):
        """ Returns vasculature radii """
        return 0.5 * self._impl.diameters

    @property
    def segment_radii(self):
        """ Returns radii for starts and ends of segments """
        edges, radii = self.edges, self.radii
        return radii[edges.T]

    @property
    def segment_points(self):
        """ Returns points for starts and ends of segments """
        points, edges = self.points, self.edges
        return points[edges.T]

    @property
    def bounding_box(self):
        """ Returns bb object """
        from archngv.spatial import BoundingBox
        return BoundingBox.from_points(self.points)

    @property
    def volume(self):
        """ Returns the total volume of the vasculature """
        from vasculatureapi.point_graph.features import segment_volumes
        return segment_volumes(self._impl).sum()

    @property
    def area(self):
        """ Returns the total area of the vasculature """
        from vasculatureapi.point_graph.features import segment_lateral_areas
        return segment_lateral_areas(self._impl).sum()

    @property
    def length(self):
        """ Returns the total length of the vasculature """
        from vasculatureapi.point_graph.features import segment_lengths
        return segment_lengths(self._impl).sum()

    def spatial_index(self):
        """ Returns vasculature spatial index """
        from spatial_index import sphere_rtree
        return sphere_rtree(self.points, self.radii)

    @property
    def point_graph(self):
        """ Returns a directed graph of the vasculature """
        return self._impl.adjacency_matrix

    @property
    def map_edges_to_sections(self):
        """ Returns section id for each edge """
        multi_index = self._impl.edge_properties.index
        return multi_index.get_level_values('section_id').to_numpy()
