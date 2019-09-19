""" Boundary stopping class
"""

import numpy as np
from scipy.spatial import cKDTree

from archngv.spatial.collision import convex_shape_with_point
from archngv.math_utils.ngons import subdivide_triangles_by_total_area


class StopAtConvexBoundary:
    """ Collision boundary function object class for microdomains.

    It checks if a point lies inside the microdomain boundary. If it does
    it doesn't stop (False). If the point lies outside the microdomain there
    is an exponential probability of stopping that depends on the distance to
    the microdomain surface.

    The closest distance to the surface of the microdomain mesh is approximated
    by distributing points on it and searching for the closest point using a KD tree.

    Args:
        points: array[float, (N, 3)]
        triangles: array[int, (M, 3)]
        triangle_normals: array[float, (M, 3)]
        hazard_rate: float
    """
    def __init__(self, points, triangles, triangle_normals, hazard_rate=0.01):

        self.face_points = points[triangles[:, 0]]
        self.face_normals = triangle_normals
        self.hazard_rate = hazard_rate

        # TODO: use a grid basis transformed onto triangle faces for a better
        # distribution of points, instead of subdivisions
        seeds, _ = \
            subdivide_triangles_by_total_area(points, triangles, len(points) * 100)

        self._seed_tree = cKDTree(seeds)

    def survival(self, distance):
        """ Exponential survival function S(d) = exp(-l*d) """
        return np.exp(-distance * self.hazard_rate)

    def acceptance_criterion(self, distance):
        """ If the distance survives then we don't collide yet.
        Cummulative F(d) = 1 - S(d) = Pr(D <= d)

        Returns: False if we stop
        """
        return not 1.0 - self.survival(distance) < np.random.random()

    def closest_point(self, point):
        """ Closest point on the surface of the convex hull """
        return self._seed_tree.query(point)

    def __call__(self, point):
        """ If the point lies inside the convex hull then it doesn't collide.
        If it is outside of it there is an exponential probability of surviving
        that goes down with the distance to the boundary
        """
        if convex_shape_with_point(self.face_points, self.face_normals, point):
            return False

        distance, _ = self.closest_point(point)
        return self.acceptance_criterion(distance)
