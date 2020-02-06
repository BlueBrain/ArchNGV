""" Data structures for endfeet meshes
"""
from collections import namedtuple
import numpy as np

from archngv.core.common import H5ContextManager


EndfootMesh = namedtuple('EndfootMesh', ['index', 'points', 'triangles', 'area', 'thickness'])


def _index_to_key(endfoot_index):
    """ Convert the endfoot index to the group key in h5 """
    return 'endfoot_' + str(endfoot_index)


def _key_to_index(key):
    return int(key.split('_')[-1])


class EndfeetAreas(H5ContextManager):
    """ Access to the endfeet meshes
    """
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
        return self._attributes['surface_areas'][index]

    def _get_mesh_surface_thickness(self, index):
        return self._attributes['surface_thicknesses'][index]

    def _object(self, endfoot_index):
        """ Returns endfoot object from its index """
        entry = self._entry(_index_to_key(endfoot_index))
        points = entry['points'][:]
        triangles = entry['triangles'][:]
        surface_area = self._get_mesh_surface_area(endfoot_index)
        surface_thickness = self._get_mesh_surface_thickness(endfoot_index)
        return EndfootMesh(endfoot_index, points, triangles, surface_area, surface_thickness)

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
        return self._entry(_index_to_key(endfoot_index))['points'][:]

    def mesh_triangles(self, endfoot_index):
        """ Return the triangles of the endfoot mesh """
        return self._entry(_index_to_key(endfoot_index))['triangles'][:]

    @property
    def mesh_surface_areas(self):
        """ Returns the surface areas of the endfeet on the vasculature """
        return self._attributes['surface_areas'][:]

    @property
    def mesh_surface_thicknesses(self):
        """ Returns the tickness of the endfeet meshes """
        return self._attributes['surface_thicknesses'][:]
