"""
Data structures for accessing information of astrocytic microdomains
"""
import os
import numpy as np

from .common import H5ContextManager


class MicrodomainTesselation(H5ContextManager):
    """ Data structure for storing the information concerning
        the microdomains.
    """

    def __init__(self, filepath):
        super(MicrodomainTesselation, self).__init__(filepath)

        self._offset_t = \
        {
            "points"    : 0,
            "triangles" : 1,
            "neighbors" : 2,
            "all"       : None,
            "domain_data" : slice(0, 1)
        }

        self._offsets = self._fd['/offsets']
        self._connectivity = self._fd['/connectivity']

        self._raw_points = self._fd['/Data/points']
        self._raw_triangles = self._fd['/Data/triangles']
        self._raw_neighbors = self._fd['/Data/neighbors']

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

    @property
    def n_microdomains(self):
        """ Number of Microdomains """
        return self.__len__()

    def n_neighbors(self, astrocyte_index):
        """ Number of neighboring microdomains around microdomains
        with astrocyte_index.
        """
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['neighbors'])
        return end - beg

    def _offset_slice(self, astrocyte_index, offset_type):
        """ Returns the slice of offset_type indices (beg, end) for astrocyte_index
        """
        return self._offsets[astrocyte_index, offset_type], \
               self._offsets[astrocyte_index + 1, offset_type]

    def domain_neighbors(self, astrocyte_index):
        """ Returns the indices of the neighboring microdomains.
        """
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['neighbors'])
        return  self._raw_neighbors[beg: end]

    def domain_points(self, astrocyte_index):
        """ The coordinates of the vertices of the microdomain.
        """
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['points'])
        return self._raw_points[beg: end]

    def domain_triangles(self, astrocyte_index):
        """ The triangles connectivity of the domain_points
        """
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['triangles'])
        return self._raw_triangles[beg: end]

    def domain_object(self, astrocyte_index):
        """ Microdomain as a ConvexPolygon object.
        """
        from morphspatial import ConvexPolygon
        return ConvexPolygon(self.domain_points(astrocyte_index),
                             self.domain_triangles(astrocyte_index))

    def iter_points(self):
        """ Iterator of the points of the vertices of each microdomain
        """
        return map(self.domain_points, range(self.n_microdomains))

    def iter_triangles(self):
        """ Iterator of the triangles of each microdomain """
        return map(self.domain_triangles, range(self.n_microdomains))

    def iter_neighbors(self):
        """ Iterator of the neighbor indices of each microdomain """
        return map(self.domain_neighbors, range(self.n_microdomains))


class MicrodomainTesselationInfo(MicrodomainTesselation):
    """ Rich data structure for storing the information concerning
        the microdomains.
    """

    def __init__(self, ngv_config):
        filepath = ngv_config.output_paths('overlapping_microdomain_structure')
        super(MicrodomainTesselationInfo, self).__init__(filepath)
        self._config = ngv_config

    def domain_mesh_path(self, astrocyte_index):
        """ Microdomain mesh path """
        return os.path.join(self._config.microdomains_directory, '{}.stl'.format(astrocyte_index))

    def domain_mesh_object(self, astrocyte_index):
        """ Microdomain mesh object """
        import stl
        return stl.Mesh.from_file(self.domain_mesh_path(astrocyte_index))
