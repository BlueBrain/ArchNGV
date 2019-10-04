"""
Data structures for accessing information of astrocytic microdomains
"""
import h5py
from cached_property import cached_property

import numpy as np
from archngv.spatial import ConvexPolygon
import archngv.core._data_microdomains as _impl


TRIANGLE_TYPE = {
    'polygon_id': 0,
    'vertices': slice(1, 4)}


OFFSET_TYPE = {
    "points": 0,
    "triangle_data": 1,
    "neighbors": 2,
    "all": None,
    "domain_data": slice(0, 2)}


class Microdomain(ConvexPolygon):
    """ Extends Convex Polygon shape into an astrocytic microdomain
    with extra properties
    """
    def __init__(self, points, triangle_data, neighbor_ids):

        self._polygon_ids = triangle_data[:, TRIANGLE_TYPE['polygon_id']]
        triangles = triangle_data[:, TRIANGLE_TYPE['vertices']]
        super(Microdomain, self).__init__(points, triangles)
        self.neighbor_ids = neighbor_ids

    @property
    def polygons(self):
        """ Returns the polygons of the domain """
        return _impl.triangles_to_polygons(self._triangles, self._polygon_ids)

    @property
    def triangle_data(self):
        """ Returns the triangle data of the microdomains
        [polygon_id, v0, v1, v2]
        """
        return np.column_stack((self._polygon_ids, self._triangles))


class MicrodomainTesselation:
    """ Data structure for storing the information concerning
        the microdomains.
    """
    def __init__(self, filepath):
        self._fd = h5py.File(filepath, 'r')

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
        return self._dset_triangle_data[:, TRIANGLE_TYPE['vertices']]

    @property
    def _polygon_ids(self):
        return self._dset_triangle_data[:, TRIANGLE_TYPE['polygon_id']]

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
        beg, end = self._offset_slice(astrocyte_index, OFFSET_TYPE['neighbors'])
        neighbors = self._dset_neighbors[beg: end]
        if omit_walls:
            return neighbors[neighbors >= 0]
        return neighbors

    def domain_points(self, astrocyte_index):
        """ The coordinates of the vertices of the microdomain. """
        beg, end = self._offset_slice(astrocyte_index, OFFSET_TYPE['points'])
        return self._dset_points[beg: end]

    def domain_triangles(self, astrocyte_index):
        """ The triangles connectivity of the domain_points
        """
        beg, end = self._offset_slice(astrocyte_index, OFFSET_TYPE['triangle_data'])
        return self._dset_triangle_data[beg: end, TRIANGLE_TYPE['vertices']]

    def domain_triangle_data(self, astrocyte_index):
        """ Returns the triangle data of the tesselation: [polygon_id, v0, v1, v2] """
        beg, end = self._offset_slice(astrocyte_index, OFFSET_TYPE['triangle_data'])
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
        ps_tris_offsets = self._offsets[:, OFFSET_TYPE['domain_data']]
        return _impl.local_to_global_mapping(self._points, self._triangles, ps_tris_offsets)

    def global_polygons(self):
        """ Returns unique points and polygons in the global index space """
        g_poly_ids = _impl.local_to_global_polygon_ids(self._polygon_ids)

        ps_tris_offsets = self._offsets[:, OFFSET_TYPE['domain_data']]

        # local to global triangles
        ps, tris, polys = _impl.local_to_global_mapping(
            self._points, self._triangles, ps_tris_offsets, triangle_labels=g_poly_ids)

        return ps, _impl.triangles_to_polygons(tris, polys)

    def export_mesh(self, filename):
        """ Write the tesselation to file as a mesh """
        import stl.mesh
        points, triangles = self.global_triangles()
        cell_mesh = stl.mesh.Mesh(np.zeros(len(triangles), dtype=stl.mesh.Mesh.dtype))
        cell_mesh.vectors = points[triangles]
        cell_mesh.save(filename)


class MicrodomainTesselationInfo(MicrodomainTesselation):
    """ Rich data structure for storing the information concerning
        the microdomains.
    """

    def __init__(self, ngv_config):
        filepath = ngv_config.output_paths('overlapping_microdomain_structure')
        super(MicrodomainTesselationInfo, self).__init__(filepath)
        self._config = ngv_config
