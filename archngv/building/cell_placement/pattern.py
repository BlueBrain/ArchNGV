"""
Handles the spatial point pattern generation and spatial indexing for the cell placement
"""

import numpy

from ngv_spatial_index import sphere_rtree


class SpatialSpherePattern:
    """ Data Structure for a sphere collection embedded in space,
    registered in an Rtree index.

    Args:
        max_spheres : int
            Maximum Number of spheres in the pattern.

    Attributes:
        coordinates: 2D array
            Coordinates of the centers of the spheres stored in the pattern
        radii: 1D array
            Respective radii
        index: int
            The current position in the coordinates / radii arrays.
        si: sphere_rtree
            The rtree spatial index data structure
    """
    def __init__(self, max_spheres):

        self._coordinates = numpy.zeros((max_spheres, 3), dtype=numpy.float)
        self._radii = numpy.zeros(max_spheres, dtype=numpy.float)

        self._index = 0

        self._si = sphere_rtree()

    def __getitem__(self, pos):
        """ Get sphere center and radius at position pos """
        assert pos < self._index
        return self._coordinates[pos], self._radii[pos]

    def __len__(self):
        """ Number of elements in the index """
        return self._index

    @property
    def coordinates(self):
        """ Returns a view of the stored coordinates """
        return self._coordinates[:self._index]

    @property
    def radii(self):
        """ Returns a view of the stored radii """
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

        Args:
            new_position: 1D array
            new_radius: float

        Returns: Bool
            True if there is intersection with another object.
        """
        return self._si.is_intersecting(new_position[0], new_position[1], new_position[2], new_radius)

    def nearest_neighbor(self, trial_position):
        """ Yields the nearest neighbor index of the sphere (new_position, new_radius)

        Args:
            trial_position: 1D array

        Returns:
            Index of the nearest neighbor.
        """
        return self._si.nearest(trial_position[0], trial_position[1], trial_position[2], 1)

    def distance_to_nearest_neighbor(self, trial_position):
        """ Distance to nearest neighbor
        """
        index = self.nearest_neighbor(trial_position)
        return numpy.linalg.norm(self.coordinates[index] - trial_position)
