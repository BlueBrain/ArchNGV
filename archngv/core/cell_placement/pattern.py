"""
Handles the spatial point pattern generation and spatial indexing for the cell placement
"""

import numpy
from spatial_index import sphere_rtree  # pylint: disable=no-name-in-module


def collision_sphere_with_sphere(point_1, radius_1, point_2, radius_2):
    """ Checks if two spheres collide by comparing their intercenter distance
    versus the sum of their radii
    """
    return numpy.sqrt((point_2[0] - point_1[0]) ** 2 + \
                      (point_2[1] - point_1[1]) ** 2 + \
                      (point_2[2] - point_1[2]) ** 2) <= radius_1 + radius_2


class SpatialSpherePattern(object):
    """ Data Structure for a sphere collection embedded in space,
    registered in an Rtree index.

    max_spheres : int
        Maximum Number of spheres in the pattern.

    """
    def __init__(self, max_spheres):

        # the centers and radii of the spheres
        self._coordinates = numpy.zeros((max_spheres, 3), dtype=numpy.float)
        self._radii = numpy.zeros(max_spheres, dtype=numpy.float)

        self._index = 0

        self._si = sphere_rtree()

    def __getitem__(self, value):
        assert value < self._index
        return self._coordinates[value], self._radii[value]

    def __len__(self):
        return self._index

    @property
    def coordinates(self):
        """ Returns a view of the stored coordinates
        """
        return self._coordinates[:self._index]

    @property
    def radii(self):
        """ Returns a view of the stored radii
        """
        return self._radii[:self._index]

    def add(self, position, radius):
        """ Add a sphere with index in the pattern and register it in the spatial index
        numpy array positional index is stored as id in RTree object
        """
        self._coordinates[self._index] = position
        self._radii[self._index] = radius

        self._si.insert(position[0], position[1], position[2], radius)
        self._index += 1

    def is_intersecting(self, new_position, new_radius):
        """ Checks if the new sphere intersects with another from
        the index. RTree intersection iterator is empty in case of
        no hits which raises a StopIteration exception.
        """
        return self._si.is_intersecting(new_position[0],
                                        new_position[1],
                                        new_position[2], new_radius)

    def nearest_neighbour(self, new_position, new_radius):
        """ Yields the nearest neighbor index of the sphere (new_position, new_radius)
        """
        return self._si.nearest(new_position[0], new_position[1], new_position[2], 1)
