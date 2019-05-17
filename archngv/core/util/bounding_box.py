""" Containts bounding box class """

import numpy as np


class BoundingBox(object):
    """ Bounding box data object"""

    @classmethod
    def from_points(cls, points):
        """ bbox constructor from point cloud """
        min_coordinates = points.min(axis=0)
        max_coordinates = points.max(axis=0)

        return cls(min_coordinates, max_coordinates)

    @classmethod
    def from_spheres(cls, points, radii):
        """ bbox constructor from spheres """
        min_coordinates = (points - radii[:, np.newaxis]).min(axis=0)
        max_coordinates = (points + radii[:, np.newaxis]).max(axis=0)

        return cls(min_coordinates, max_coordinates)

    def __init__(self, min_coordinates, max_coordinates):

        self._bb = np.array((min_coordinates,
                             max_coordinates), dtype=np.float)

    def __eq__(self, other):
        """ Equality of bboxes"""
        return np.allclose(self.min_point, other.min_point) and \
               np.allclose(self.max_point, other.max_point)

    def __add__(self, other):
        """ Create a new bbox that is the support of both bboxes"""
        return BoundingBox(np.min((self.min_point, other.min_point), axis=0),
                           np.max((self.max_point, other.max_point), axis=0))

    @property
    def ranges(self):
        """ x_range, y_range, z_range """
        return self._bb

    @property
    def offset(self):
        """ Get the offset from the origin """
        return self._bb[0]

    @property
    def min_point(self):
        """ Return the min point """
        return self.offset

    @property
    def max_point(self):
        """ Return the max point """
        return self._bb[1]

    @property
    def center(self):
        """ Center of bounding box """
        return 0.5 * (self.min_point + self.max_point)

    @property
    def extent(self):
        """ Get the difference between the max and min range per dimension """
        return np.array(((self._bb[1][0] - self._bb[0][0]),
                         (self._bb[1][1] - self._bb[0][1]),
                         (self._bb[1][2] - self._bb[0][2])))

    @property
    def layout(self):
        """ Get the points and the respective edges of the bounding box """
        (xmin, ymin, zmin, xmax, ymax, zmax) = self._bb.ravel()

        points = np.array(((xmin, ymin, zmin),
                           (xmin, ymax, zmin),
                           (xmin, ymin, zmax),
                           (xmin, ymax, zmax),
                           (xmax, ymin, zmin),
                           (xmax, ymax, zmin),
                           (xmax, ymin, zmax),
                           (xmax, ymax, zmax)))

        edges = np.array(((0, 1), (0, 2), (0, 4), (1, 3),
                          (1, 5), (2, 3), (2, 6), (3, 7),
                          (4, 5), (4, 6), (5, 7), (6, 7)))

        return points, edges

    @property
    def volume(self):
        """ Volume of bbox """
        xmin, ymin, zmin = self.min_point
        xmax, ymax, zmax = self.max_point

        a = np.array([xmax - xmin, 0, 0])
        b = np.array([0, ymax - ymin, 0])
        c = np.array([0, 0, zmax - zmin])

        return np.abs(np.dot(np.cross(a, b), c))

    def translate(self, point):
        """ Translate by point coordinates """
        self._bb += point

    def scale(self, triplet):
        """ Scale bbox """
        center = self.center
        self.translate(-center)
        self._bb *= triplet
        self.translate(center)

    def points_inside(self, points):
        """ Returns a boolean mask of the points included in the
        bounding box
        """
        return np.all(np.logical_and(self.min_point <= points, points <= self.max_point), axis=1)

    def spheres_inside(self, centers, radii):
        """ Returns a boolean mask of the spheres that are included in the
        bounding box
        """
        radii_expanded = radii[:, np.newaxis]
        return np.all(np.logical_and(self.min_point <= centers - radii_expanded,
                                     centers + radii_expanded <= self.max_point), axis=1)
