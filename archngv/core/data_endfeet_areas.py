""" Data structures for endfeet meshes
"""
import numpy as np

from archngv.core.common import H5ContextManager
from archngv.utils.ngons import vectorized_triangle_area


class EndfootMesh:
    """ Endfoot mesh data object

    Attrs:
        index: int,
        points: array[float, (N, 3)]
        triangles: array[float, (M, 3)]
    """
    def __init__(self, index, points, triangles, thickness):
        self.index = index
        self.points = points
        self.triangles = triangles
        self._thickness = thickness

    def __eq__(self, other):
        """ Check if two objects are equal by comparing
        their index """
        return self.index == other.index

    def __str__(self):
        """ String repr of the object """
        return 'endfoot_mesh<Index: {}>'.format(self.index)

    @property
    def thickness(self):
        """ Returns thickness of the surface """
        return self._thickness

    @property
    def area(self):
        """ Returns the area of the surface """
        side1_vecs = self.points[self.triangles[:, 1]] - self.points[self.triangles[:, 0]]
        side2_vecs = self.points[self.triangles[:, 2]] - self.points[self.triangles[:, 0]]
        return vectorized_triangle_area(side1_vecs, side2_vecs).sum()


def _index_to_key(endfoot_index):
    """ Convert the endfoot index to the group key in h5 """
    return 'endfoot_' + str(endfoot_index)


class EndfeetAreas(H5ContextManager):
    """ Access to the endfeet meshes
    """
    @property
    def _groups(self):
        """ Groups storing endfoot information """
        return self._fd['objects']

    def __len__(self):
        """ Number of endfeet """
        return len(self._groups)

    def _entry(self, endfoot_key):
        """ Return the group entry given the key """
        return self._groups[endfoot_key]

    def _object(self, endfoot_index):
        """ Returns endfoot object from its index """
        entry = self._entry(_index_to_key(endfoot_index))
        points = entry['points'][:]
        triangles = entry['triangles'][:]
        thickness = entry.attrs['thickness']
        return EndfootMesh(endfoot_index, points, triangles, thickness)

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
        return self._entry(_index_to_key(endfoot_index))['points'][:]

    def mesh_triangles(self, endfoot_index):
        """ Return the triangles of the endfoot mesh """
        return self._entry(_index_to_key(endfoot_index))['triangles'][:]
