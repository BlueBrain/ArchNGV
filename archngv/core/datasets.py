""" Archngv dataset classes """
import os
from collections import namedtuple
from cached_property import cached_property

import numpy as np
import libsonata
from archngv.core.common import H5ContextManager, EdgesContextManager

from archngv.spatial import ConvexPolygon
from archngv.core import _impl_microdomains


class Vasculature:
    """ Vasculature wrapper using VasculatureAPI
    """
    def __init__(self, vasculature):
        self._impl = vasculature

    @classmethod
    def load(cls, filepath):
        """ Load vasculature file """
        from vasculatureapi import SectionVasculature
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


class CellData(H5ContextManager):
    """ Data structure for the collection of cell characteristics. Only the actual
    file is required for accessing the respetive data with this class. No relative path
    data is available from this entry point.
    """
    def __init__(self, filepath_or_config):

        # TODO: remove the config related code when snapCircuit replaces NGVCircuit
        if isinstance(filepath_or_config, str):
            self._config = None
            filepath = filepath_or_config
        else:
            self._config = filepath_or_config
            filepath = self._config.output_paths('cell_data')

        super().__init__(filepath)
        self.astrocyte_positions = self._fd['/positions']
        self.astrocyte_radii = self._fd['/radii']

        self.astrocyte_gids = self._fd['/ids']
        self.astrocyte_names = self._fd['/names']

    def __len__(self):
        "return cell data size"
        return len(self.astrocyte_positions)

    @property
    def astrocyte_point_data(self):
        """ Returns stacked astrocyte positions and radii
        """
        return np.column_stack((self.astrocyte_positions, self.astrocyte_radii))

    @property
    def n_cells(self):
        """ Number of cells """
        return self.__len__()

    @property
    def positions(self):
        """ Returns positions """
        return self.astrocyte_positions

    @property
    def radii(self):
        """ Returns radii """
        return self.astrocyte_radii

    @property
    def names(self):
        """ Returns the cell names """
        return self.astrocyte_names

    @property
    def ids(self):
        """ Returns the astrocyte ids """
        return self.astrocyte_gids

    def morphology_path(self, astrocyte_index):
        """ Absolute path to the astrocyte morphology corresponding
        to the given index.
        """
        cell_filename = self.astrocyte_names[astrocyte_index].decode('utf-8') + '.h5'
        return os.path.join(self._config.morphology_directory, cell_filename)

    def morphology_object(self, astrocyte_index):
        """ Readonly morphology object using morphio
        Returns:
            A morphio read only object

        Notes:
            You need to pip install archngv[core] or archngv[all] to have access to this feature
        """
        from morphio import Morphology
        return Morphology(self.morphology_path(astrocyte_index))


class Microdomain(ConvexPolygon):
    """ Extends Convex Polygon shape into an astrocytic microdomain with extra properties
    """
    def __init__(self, points, triangle_data, neighbor_ids):

        self._polygon_ids = triangle_data[:, _impl_microdomains.TRIANGLE_TYPE['polygon_id']]
        triangles = triangle_data[:, _impl_microdomains.TRIANGLE_TYPE['vertices']]
        super(Microdomain, self).__init__(points, triangles)
        self.neighbor_ids = neighbor_ids

    @property
    def polygons(self):
        """ Returns the polygons of the domain """
        return _impl_microdomains.triangles_to_polygons(self._triangles, self._polygon_ids)

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


class MicrodomainTesselation(H5ContextManager):
    """ Data structure for storing the information concerning
        the microdomains.
    """
    def __init__(self, filepath):
        super().__init__(filepath)
        self._offsets = self._fd['/offsets']
        self._dset_points = self._fd['/data/points']
        self._dset_neighbors = self._fd['/data/neighbors']
        self._dset_triangle_data = self._fd['/data/triangle_data']

    def __iter__(self):
        """ Microdomain object iterator """
        for i in range(self.n_microdomains):
            yield self.domain_object(i)

    def __len__(self):
        """ Number of Microdomains """
        return len(self._offsets) - 1

    def __getitem__(self, key):
        """ list getter """
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
        return self._dset_triangle_data[:, _impl_microdomains.TRIANGLE_TYPE['vertices']]

    @property
    def _polygon_ids(self):
        return self._dset_triangle_data[:, _impl_microdomains.TRIANGLE_TYPE['polygon_id']]

    @property
    def n_microdomains(self):
        """ Number of Microdomains """
        return self.__len__()

    def n_neighbors(self, astrocyte_index, omit_walls=True):
        """ Number of neighboring microdomains around microdomains
        with astrocyte_index.
        """
        return len(self.domain_neighbors(astrocyte_index, omit_walls=omit_walls))

    def domain_neighbors(self, astrocyte_index, omit_walls=True):
        """ For every triangle in the microdomain return its respective neighbor.
        Multiple triangles can have the same neighbor if the are part of a triangulated
        polygon face. A microdomain can also have a bounding box wall as a neighbor
        which is signified with a negative number.
        """
        beg, end = self._offset_slice(astrocyte_index, _impl_microdomains.OFFSET_TYPE['neighbors'])
        neighbors = self._dset_neighbors[beg: end]
        if omit_walls:
            return neighbors[neighbors >= 0]
        return neighbors

    def domain_points(self, astrocyte_index):
        """ The coordinates of the vertices of the microdomain. """
        beg, end = self._offset_slice(astrocyte_index, _impl_microdomains.OFFSET_TYPE['points'])
        return self._dset_points[beg: end]

    def domain_triangles(self, astrocyte_index):
        """ The triangles connectivity of the domain_points
        """
        beg, end = self._offset_slice(astrocyte_index, _impl_microdomains.OFFSET_TYPE['triangle_data'])
        return self._dset_triangle_data[beg: end, _impl_microdomains.TRIANGLE_TYPE['vertices']]

    def domain_triangle_data(self, astrocyte_index):
        """ Returns the triangle data of the tesselation: [polygon_id, v0, v1, v2] """
        beg, end = self._offset_slice(astrocyte_index, _impl_microdomains.OFFSET_TYPE['triangle_data'])
        return self._dset_triangle_data[beg: end]

    def domain_object(self, astrocyte_index):
        """ Returns a Microdomain object """
        return Microdomain(self.domain_points(astrocyte_index),
                           self.domain_triangle_data(astrocyte_index),
                           self.domain_neighbors(astrocyte_index, omit_walls=False))

    @cached_property
    def connectivity(self):
        """ Returns the connectivity of the microdomains """
        edges = [(cid, nid) for cid in range(self.n_microdomains)
                            for nid in self.domain_neighbors(cid, omit_walls=True)]
        # sort by column [2 3 1] -> [1 2 3]
        sorted_by_column = np.sort(edges, axis=1)
        # take the unique rows
        return np.unique(sorted_by_column, axis=0)

    def global_triangles(self):
        """
        Converts the per object tesselation to a joined mesh with unique
        points and triangles of unique vertices.

        Args:
            unique_wall_face_id: bool
            If True every face that touches a boundary wall will be given a
            unique negative id. This is useful in order to allow for separation
            of the wall faces in the global reference system.

        Returns:
            points: array[float, (N, 3)]
            triangles: array[int, (M, 3)]
            neighbor_ids: array[int, (M, 3)]
        """
        ps_tris_offsets = self._offsets[:, _impl_microdomains.OFFSET_TYPE['domain_data']]
        return _impl_microdomains.local_to_global_mapping(self._points, self._triangles, ps_tris_offsets)

    def global_polygons(self):
        """ Returns unique points and polygons in the global index space """
        g_poly_ids = _impl_microdomains.local_to_global_polygon_ids(self._polygon_ids)

        ps_tris_offsets = self._offsets[:, _impl_microdomains.OFFSET_TYPE['domain_data']]

        # local to global triangles
        ps, tris, polys = _impl_microdomains.local_to_global_mapping(
            self._points, self._triangles, ps_tris_offsets, triangle_labels=g_poly_ids)

        return ps, _impl_microdomains.triangles_to_polygons(tris, polys)

    def export_mesh(self, filename):
        """ Write the tesselation to file as a mesh """
        import stl.mesh
        points, triangles = self.global_triangles()
        cell_mesh = stl.mesh.Mesh(np.zeros(len(triangles), dtype=stl.mesh.Mesh.dtype))
        cell_mesh.vectors = points[triangles]
        cell_mesh.save(filename)


class GliovascularData(H5ContextManager):
    """ Provides access to the endfeet contact points

    Attributes:
        endfoot_graph_coordinates: array[float, (N , 3)]
            Astrocytic endfeet connection point on the skeleton
            of the vasculature.
        endfoot_surface_coordinates: array[float, (N, 3)]
            Astrocytic endfeet connection points on the surface
            of the vasculature.
    """
    def __init__(self, filepath):
        super().__init__(filepath)

        self.endfoot_graph_coordinates = \
            self._fd['/endfoot_graph_coordinates']

        self.endfoot_surface_coordinates = \
            self._fd['/endfoot_surface_coordinates']

    @property
    def n_endfeet(self):
        """ Total number of endfeet """
        return len(self.endfoot_graph_coordinates)

    @property
    def vasculature_surface_targets(self):
        return self.endfoot_surface_coordinates[:]

    @property
    def vasculature_skeleton_targets(self):
        return self.endfoot_graph_coordinates[:]

class SynapticData(EdgesContextManager):
    """ Synaptic data access """
    def _select(self, synapse_ids):
        if synapse_ids is None:
            return libsonata.Selection([(0, self._impl.size)])
        else:
            return libsonata.Selection(synapse_ids)

    def synapse_coordinates(self, synapse_ids=None):
        """ XYZ coordinates for given synapse_ids (all if synapse_ids not specified) """
        selection = self._select(synapse_ids)

        try:
            return np.stack([
                self._impl.get_attribute('efferent_center_x', selection),
                self._impl.get_attribute('efferent_center_y', selection),
                self._impl.get_attribute('efferent_center_z', selection),
            ]).transpose()
        except libsonata.SonataError:
            return np.stack([
                self._impl.get_attribute('afferent_center_x', selection),
                self._impl.get_attribute('afferent_center_y', selection),
                self._impl.get_attribute('afferent_center_z', selection),
            ]).transpose()

    def afferent_gids(self, synapse_ids=None):
        """ 0-based afferent neuron GIDs for given synapse_ids (all if synapse_ids not specified) """
        selection = self._select(synapse_ids)
        return self._impl.target_nodes(selection)

    @cached_property
    def n_neurons(self):
        """ Number of afferent neurons """
        return 1 + np.max(self.afferent_gids()).astype(int)  # TODO: get from HDF5 attributes

    @property
    def n_synapses(self):
        """ Number of synapses """
        return self._impl.size


class EndfeetAreas(H5ContextManager):
    """ Access to the endfeet meshes
    """
    EndfootMesh = namedtuple('EndfootMesh', ['index', 'points', 'triangles', 'area', 'thickness'])

    @staticmethod
    def _index_to_key(endfoot_index):
        """ Convert the endfoot index to the group key in h5 """
        return 'endfoot_' + str(endfoot_index)

    @staticmethod
    def _key_to_index(key):
        return int(key.split('_')[-1])

    @property
    def _groups(self):
        """ Groups storing endfoot information """
        return self._fd['objects']

    @property
    def _attributes(self):
        """ Group storing properties datasets"""
        return self._fd['attributes']

    def __len__(self):
        """ Number of endfeet """
        return len(self._groups)

    def _entry(self, endfoot_key):
        """ Return the group entry given the key """
        return self._groups[endfoot_key]

    def _get_mesh_surface_area(self, index):
        return self._attributes['surface_area'][index]

    def _get_mesh_surface_thickness(self, index):
        return self._attributes['surface_thickness'][index]

    def _object(self, endfoot_index):
        """ Returns endfoot object from its index """
        entry = self._entry(self._index_to_key(endfoot_index))
        points = entry['points'][:]
        triangles = entry['triangles'][:]
        surface_area = self._get_mesh_surface_area(endfoot_index)
        surface_thickness = self._get_mesh_surface_thickness(endfoot_index)
        return self.EndfootMesh(endfoot_index, points, triangles, surface_area, surface_thickness)

    def __iter__(self):
        """ Endfoot iterator """
        for index in range(self.__len__()):
            yield self._object(index)

    def __getitem__(self, index):
        """ Endfoot mesh object getter """
        if isinstance(index, (np.integer, int)):
            return self._object(index)
        if isinstance(index, slice):
            return [self._object(i) for i in range(*index.indices(len(self)))]
        if isinstance(index, np.ndarray):
            return [self._object(i) for i in index]
        raise TypeError("Invalid argument type: ({}, {})".format(type(index), index))

    def mesh_points(self, endfoot_index):
        """ Return the points of the endfoot mesh """
        return self._entry(self._index_to_key(endfoot_index))['points'][:]

    def mesh_triangles(self, endfoot_index):
        """ Return the triangles of the endfoot mesh """
        return self._entry(self._index_to_key(endfoot_index))['triangles'][:]

    def get(self, attribute_name, ids=None):
        """ Get the respective attribute array """
        dset = self._attributes[attribute_name][:]
        if ids is not None:
            return dset[ids]
        return dset
