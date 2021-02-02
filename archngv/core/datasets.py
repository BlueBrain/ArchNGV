"""Archngv dataset classes."""

from collections import namedtuple
from cached_property import cached_property

import numpy as np
import h5py

from archngv.exceptions import NGVError
from archngv.core.sonata_readers import EdgesReader, NodesReader

from archngv.spatial import ConvexPolygon


DOMAIN_TRIANGLE_TYPE = {
    'polygon_id': 0,
    'vertices': slice(1, 4)}


DOMAIN_OFFSET_TYPE = {
    "points": 0,
    "triangle_data": 1,
    "neighbors": 2,
    "all": None,
    "domain_data": slice(0, 2)}


class CellData(NodesReader):
    """Cell population information"""

    def positions(self, index=None):
        """Cell positions"""
        return self.get_properties(['x', 'y', 'z'], ids=index)


class GliovascularConnectivity(EdgesReader):
    """Access to the Gliovascular Data"""

    def astrocyte_endfeet(self, astrocyte_ids):
        """endfoot_id is equivalent to the edge id. Can resolve quicker using afferent_edges"""
        return self.afferent_edges(astrocyte_ids)

    def vasculature_surface_targets(self, endfeet_ids=None):
        """Endfeet surface targets on vasculature."""
        return self.get_properties(
            ['endfoot_surface_x', 'endfoot_surface_y', 'endfoot_surface_z'], ids=endfeet_ids)

    def vasculature_sections_segments(self, endfeet_ids):
        """Returns the edge id, morphology section and segment id for each endfoot"""
        edge_ids = self.get_source_nodes(endfeet_ids)
        efferent = self.get_properties(['vasculature_section_id', 'vasculature_segment_id'], ids=endfeet_ids)
        return np.column_stack((edge_ids, efferent))


class NeuronalConnectivity(EdgesReader):
    """ Synaptic data access """

    def synapse_positions(self, synapse_ids=None):
        """ XYZ coordinates for given synapse_ids (all if synapse_ids not specified) """
        syn_positions = [['efferent_center_x', 'efferent_center_y', 'efferent_center_z'],
                         ['afferent_center_x', 'afferent_center_y', 'afferent_center_z']]

        for position_properties in syn_positions:
            try:
                return self.get_properties(position_properties, synapse_ids)
            except NGVError:
                pass

        raise NGVError(f"Cannot find positions inside {self.filepath}")

    def target_neurons(self, synapse_ids=None):
        """Target neuron's node ids for given synapse_ids."""
        return self._impl.target_nodes(self._selection(synapse_ids))

    @cached_property
    def target_neuron_count(self):
        """Number of unique target neurons."""
        return len(np.unique(self.target_neurons()))


class NeuroglialConnectivity(EdgesReader):
    """Neuroglial connectivity access."""

    def astrocyte_neuron_connections(self, astrocyte_id):
        """Returns edge ids between astrocyte and neurons"""
        return self.efferent_edges(astrocyte_id)

    def neuronal_synapses(self, connection_ids):
        """Returns the synapse ids"""
        return self.get_property('synapse_id', ids=connection_ids)

    def astrocyte_synapses(self, astrocyte_id):
        """Get the synapse ids connected to a given `astrocyte_id`."""
        edge_ids = self.astrocyte_neuron_connections(astrocyte_id)
        return self.neuronal_synapses(edge_ids)

    def astrocyte_number_of_synapses(self, astrocyte_id):
        """Get the number of synapses to a given `astrocyte_id`."""
        return len(np.unique(self.astrocyte_synapses(astrocyte_id)))

    def astrocyte_neurons(self, astrocyte_id, unique=True):
        """Post-synaptic neurons given an `astrocyte_id`."""
        return self.efferent_nodes(astrocyte_id, unique=unique)


class GlialglialConnectivity(EdgesReader):
    """Glialglial connectivity access."""

    def astrocyte_astrocytes(self, astrocyte_id, unique=True):
        """Target astrocyte connected to astrocyte with `astrocyte_id`."""
        return self.efferent_nodes(astrocyte_id, unique=unique)


class Vasculature:
    """ Vasculature wrapper using VasculatureAPI"""
    def __init__(self, vasculature):
        from vasculatureapi import PointVasculature
        if isinstance(vasculature, PointVasculature):
            self._impl = vasculature
        else:
            raise NGVError("Vasculature init with something else than PointVasculature")

    @classmethod
    def load(cls, filepath):
        """ Load vasculature file """
        from vasculatureapi import SectionVasculature
        section_vasculature = SectionVasculature.load(filepath)
        point_vasculature = section_vasculature.as_point_graph()
        return cls(point_vasculature)

    @classmethod
    def load_sonata(cls, filepath):
        """Loads a vasculature SONATA NodePopulation"""
        from archngv.building.exporters.node_populations import load_vasculature_node_population
        return cls(load_vasculature_node_population(filepath))

    def save_sonata(self, filepath):
        """Writes the vasculature dataset as a SONATA NodePopulation
        """
        from archngv.building.exporters.node_populations import save_vasculature_node_population
        save_vasculature_node_population(self._impl, filepath)

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
        return self.radii[self.edges.T]

    @property
    def segment_points(self):
        """ Returns points for starts and ends of segments """
        return self.points[self.edges.T]

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

    @property
    def point_graph(self):
        """ Returns a directed graph of the vasculature """
        return self._impl.adjacency_matrix

    @property
    def map_edges_to_sections(self):
        """ Returns section id for each edge """
        multi_index = self._impl.edge_properties.index
        return multi_index.get_level_values('section_id').to_numpy()


class Microdomain(ConvexPolygon):
    """ Extends Convex Polygon shape into an astrocytic microdomain with extra properties
    """
    def __init__(self, points, triangle_data, neighbor_ids):
        self._polygon_ids = triangle_data[:, DOMAIN_TRIANGLE_TYPE['polygon_id']]
        triangles = triangle_data[:, DOMAIN_TRIANGLE_TYPE['vertices']]
        super().__init__(points, triangles)
        self.neighbor_ids = neighbor_ids

    @property
    def polygons(self):
        """ Returns the polygons of the domain """
        from archngv.utils.ngons import triangles_to_polygons
        return triangles_to_polygons(self._triangles, self._polygon_ids)

    @property
    def triangle_data(self):
        """ Returns the triangle data of the microdomains
        [polygon_id, v0, v1, v2]
        """
        return np.column_stack((self._polygon_ids, self._triangles))

    def scale(self, scale_factor):
        """ Uniformly scales the polygon by a scale_factor, assuming its centroid
        sits on the origin.
        """
        cnt = self.centroid
        return Microdomain(
            scale_factor * (self.points - cnt) + cnt,
            self.triangle_data.copy(),
            self.neighbor_ids.copy())


class H5ContextManager:
    """ Context manager for hdf5 files """

    def __init__(self, filepath):
        self._fd = h5py.File(filepath, 'r')

    def close(self):
        """ Close hdf5 file """
        self._fd.close()

    def __enter__(self):
        """ Context manager entry """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ And exit """
        self.close()


class MicrodomainTesselation(H5ContextManager):
    """Data structure for storing the information concerning the microdomains."""
    def __init__(self, filepath):
        super().__init__(filepath)
        self._offsets = self._fd['/offsets']
        self._dset_points = self._fd['/data/points']
        self._dset_neighbors = self._fd['/data/neighbors']
        self._dset_triangle_data = self._fd['/data/triangle_data']

    def __iter__(self):
        """Microdomain object iterator."""
        for i in range(self.n_microdomains):
            yield self.domain_object(i)

    def __len__(self):
        """Number of Microdomains."""
        return len(self._offsets) - 1

    def __getitem__(self, key):
        """List getter."""
        if isinstance(key, slice):
            return [self.domain_object(i) for i in range(key.start, key.stop, key.step)]
        elif isinstance(key, (np.integer, int)):
            return self.domain_object(key)
        else:
            raise TypeError("Invalid argument type: ({}, {})".format(type(key), key))

    def _offset_slice(self, astrocyte_index, offset_type):
        """ Returns the slice of offset_type indices (beg, end) for astrocyte_index
        """
        return self._offsets[astrocyte_index, offset_type], \
               self._offsets[astrocyte_index + 1, offset_type]

    @property
    def _points(self):
        return self._dset_points[:]

    @property
    def _triangles(self):
        return self._dset_triangle_data[:, DOMAIN_TRIANGLE_TYPE['vertices']]

    @property
    def _polygon_ids(self):
        return self._dset_triangle_data[:, DOMAIN_TRIANGLE_TYPE['polygon_id']]

    @property
    def n_microdomains(self):
        """Total number of Microdomains."""
        return self.__len__()

    def n_neighbors(self, astrocyte_index, omit_walls=True):
        """Number of neighboring microdomains around microdomains using astrocyte_index."""
        return len(self.domain_neighbors(astrocyte_index, omit_walls=omit_walls))

    def domain_neighbors(self, astrocyte_index, omit_walls=True):
        """For every triangle in the microdomain return its respective neighbor.

        Multiple triangles can have the same neighbor if the are part of a triangulated
        polygon face. A microdomain can also have a bounding box wall as a neighbor
        which is signified with a negative number.
        """
        beg, end = self._offset_slice(astrocyte_index, DOMAIN_OFFSET_TYPE['neighbors'])
        neighbors = self._dset_neighbors[beg: end]
        if omit_walls:
            return neighbors[neighbors >= 0]
        return neighbors

    def domain_is_boundary(self, astrocyte_index):
        """Returns true if the domain is adjacent to a wall."""
        beg, end = self._offset_slice(astrocyte_index, DOMAIN_OFFSET_TYPE['neighbors'])
        return np.any(self._dset_neighbors[beg: end] < 0)

    def domain_points(self, astrocyte_index):
        """The coordinates of the vertices of the microdomain."""
        beg, end = self._offset_slice(astrocyte_index, DOMAIN_OFFSET_TYPE['points'])
        return self._dset_points[beg: end]

    def domain_triangles(self, astrocyte_index):
        """Returns the triangles connectivity of the domain_points from an astrocyte."""
        beg, end = self._offset_slice(astrocyte_index, DOMAIN_OFFSET_TYPE['triangle_data'])
        return self._dset_triangle_data[beg: end, DOMAIN_TRIANGLE_TYPE['vertices']]

    def domain_triangle_data(self, astrocyte_index):
        """Returns the triangle data of the tesselation.

        Returns:
            numpy.ndarray: [polygon_id, v0, v1, v2]
        """
        beg, end = self._offset_slice(astrocyte_index, DOMAIN_OFFSET_TYPE['triangle_data'])
        return self._dset_triangle_data[beg: end]

    def domain_object(self, astrocyte_index):
        """Returns a archngv.core.dataset Microdomain object."""
        return Microdomain(self.domain_points(astrocyte_index),
                           self.domain_triangle_data(astrocyte_index),
                           self.domain_neighbors(astrocyte_index, omit_walls=False))

    @cached_property
    def connectivity(self):
        """Returns the connectivity of the microdomains."""
        edges = [(cid, nid) for cid in range(self.n_microdomains)
                            for nid in self.domain_neighbors(cid, omit_walls=True)]
        # sort by column [2 3 1] -> [1 2 3]
        sorted_by_column = np.sort(edges, axis=1)
        # take the unique rows
        return np.unique(sorted_by_column, axis=0)

    def global_triangles(self):
        """Converts microdomain tesselation to a joined mesh.

        Converts the per object tesselation to a joined mesh with unique points and triangles
        of unique vertices.

        Returns:
            points: array[float, (N, 3)]
            triangles: array[int, (M, 3)]
            neighbor_ids: array[int, (M, 3)]
        """
        from archngv.utils.ngons import local_to_global_mapping
        ps_tris_offsets = self._offsets[:, DOMAIN_OFFSET_TYPE['domain_data']]
        return local_to_global_mapping(self._points, self._triangles, ps_tris_offsets)

    def global_polygons(self):
        """Returns unique points and polygons in the global index space."""
        from archngv.utils.ngons import triangles_to_polygons
        from archngv.utils.ngons import local_to_global_mapping
        from archngv.utils.ngons import local_to_global_polygon_ids
        g_poly_ids = local_to_global_polygon_ids(self._polygon_ids)

        ps_tris_offsets = self._offsets[:, DOMAIN_OFFSET_TYPE['domain_data']]

        # local to global triangles
        ps, tris, polys = local_to_global_mapping(
            self._points, self._triangles, ps_tris_offsets, triangle_labels=g_poly_ids)

        return ps, triangles_to_polygons(tris, polys)

    def export_mesh(self, filename):
        """Write the tesselation to file as a mesh."""
        import stl.mesh
        points, triangles = self.global_triangles()
        cell_mesh = stl.mesh.Mesh(np.zeros(len(triangles), dtype=stl.mesh.Mesh.dtype))
        cell_mesh.vectors = points[triangles]
        cell_mesh.save(filename)


EndfootMesh = namedtuple('EndfootMesh', ['index', 'points', 'triangles', 'area', 'thickness'])


class EndfootSurfaceMeshes(H5ContextManager):
    """Access to the endfeet meshes."""

    @staticmethod
    def _index_to_key(endfoot_index):
        """Convert the endfoot index to the group key in h5."""
        return 'endfoot_' + str(endfoot_index)

    @staticmethod
    def _key_to_index(key):
        return int(key.split('_')[-1])

    @property
    def _groups(self):
        """Groups storing endfoot information."""
        return self._fd['objects']

    @property
    def _attributes(self):
        """Group storing properties datasets."""
        return self._fd['attributes']

    def __len__(self):
        """Number of endfeet."""
        return len(self._groups)

    def _entry(self, endfoot_key):
        """Return the group entry given the key."""
        return self._groups[endfoot_key]

    def _get_mesh_surface_area(self, index):
        return self._attributes['surface_area'][index]

    def _get_mesh_surface_thickness(self, index):
        return self._attributes['surface_thickness'][index]

    def _object(self, endfoot_index):
        """Returns endfoot object from its index."""
        entry = self._entry(self._index_to_key(endfoot_index))
        points = entry['points'][:]
        triangles = entry['triangles'][:]
        surface_area = self._get_mesh_surface_area(endfoot_index)
        surface_thickness = self._get_mesh_surface_thickness(endfoot_index)
        return EndfootMesh(endfoot_index, points, triangles, surface_area, surface_thickness)

    def __iter__(self):
        """Endfoot iterator."""
        for index in range(self.__len__()):
            yield self._object(index)

    def __getitem__(self, index):
        """Endfoot mesh object getter."""
        if isinstance(index, (np.integer, int)):
            return self._object(index)
        if isinstance(index, slice):
            return [self._object(i) for i in range(*index.indices(len(self)))]
        if isinstance(index, np.ndarray):
            return [self._object(i) for i in index]
        raise TypeError("Invalid argument type: ({}, {})".format(type(index), index))

    def mesh_points(self, endfoot_index):
        """Return the points of the endfoot mesh."""
        return self._entry(self._index_to_key(endfoot_index))['points'][:]

    def mesh_triangles(self, endfoot_index):
        """Return the triangles of the endfoot mesh."""
        return self._entry(self._index_to_key(endfoot_index))['triangles'][:]

    def get(self, attribute_name, ids=None):
        """Get the respective attribute array."""
        dset = self._attributes[attribute_name][:]
        if ids is not None:
            return dset[ids]
        return dset
