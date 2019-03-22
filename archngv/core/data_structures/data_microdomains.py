import os
import h5py
import logging

import numpy as np

from builtins import range
from builtins import map

L = logging.getLogger(__name__)


class H5ContextManager(object):

    def __init__(self, filepath):
        self._fd = h5py.File(filepath, 'r')

    def close(self):
        self._fd.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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

        self._offsets        = self._fd['/offsets']
        self._connectivity   = self._fd['/connectivity']

        self._raw_points    = self._fd['/Data/points']
        self._raw_triangles = self._fd['/Data/triangles']
        self._raw_neighbors = self._fd['/Data/neighbors']

    def __iter__(self):
        for i in range(self.n_microdomains):
            yield self.domain_object(i)

    @property
    def n_microdomains(self):
        return len(self._offsets) - 1

    def n_neighbors(self, astrocyte_index):
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['neighbors'])
        return end - beg

    def _offset_slice(self, astrocyte_index, offset_type):
        return self._offsets[astrocyte_index, offset_type], \
               self._offsets[astrocyte_index + 1, offset_type]

    def domain_neighbors(self, astrocyte_index):
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['neighbors'])
        return  self._raw_neighbors[beg: end]

    def domain_points(self, astrocyte_index):
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['points'])
        return self._raw_points[beg: end]

    def domain_triangles(self, astrocyte_index):
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['triangles'])
        return self._raw_triangles[beg: end]

    def domain_object(self, astrocyte_index):
        from morphspatial import ConvexPolygon
        return ConvexPolygon(self.domain_points(astrocyte_index),
                             self.domain_triangles(astrocyte_index))

    def iter_points(self):
        return map(domain_points, range(self.n_microdomains))

    def iter_triangles(self):
        return map(domain_triangles, range(self.n_microdomains))

    def iter_neighbors(self):
        return map(domain_neighbors, range(self.n_microdomains))


class MicrodomainTesselationInfo(MicrodomainTesselation):

    def __init__(self, ngv_config):
        filepath = ngv_config.output_paths('overlapping_microdomain_structure')
        super(MicrodomainTesselationInfo, self).__init__(filepath)
        self._config = ngv_config

    def domain_mesh_path(self, astrocyte_index):
        return os.path.join(self._config.microdomains_directory, '{}.stl'.format(astrocyte_index))

    def domain_mesh_object(self, astrocyte_index):
        import stl
        return stl.Mesh.from_file(self.domain_mesh_path(astrocyte_index))
