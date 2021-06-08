""" Boundary stopping class
"""

import numpy as np
from scipy.spatial import cKDTree

from archngv.spatial.collision import convex_shape_with_point
from archngv.utils.ngons import subdivide_triangles_by_total_area


class StopAtConvexBoundary:
    """ Collision boundary function object class for microdomains.

    It checks if a point lies inside the microdomain boundary. If it does
    it doesn't stop (False). If the point lies outside the microdomain there
    is an exponential probability of stopping that depends on the distance to
    the microdomain surface.

    The closest distance to the surface of the microdomain mesh is approximated
    by distributing points on it and searching for the closest point using a KD tree.

    Args:
        points (np.ndarray): Array of 3D points, the vertices of the microdomain
        triangles (np.ndarray): Array of triangles, the faces of the microdomain
        triangle_normals (np.ndarra): Array of normals, one per microdomain face
        hazard_rate(float)
        rng (RandomState, Generator): Random generator to use

    Attrs:
        face_points (np.ndarray): Microdomain face points
        face_normals (np.ndarray): Microdomain face normals
        hazard_rate (float)

    Notes:
        Seeds are distributed on the faces of the microdomain depending, on its
        surface area. This seeds are used for the calculation of closest distance
        to the domain. There are generated 100 x domain vertices
    """
    def __init__(self, points, triangles, triangle_normals, hazard_rate, rng):

        self.face_points = points[triangles[:, 0]]
        self.face_normals = triangle_normals
        self.hazard_rate = hazard_rate

        # TODO: use a grid basis transformed onto triangle faces for a better
        # distribution of points, instead of subdivisions
        seeds, _ = \
            subdivide_triangles_by_total_area(points, triangles, len(points) * 100)

        self._seed_tree = cKDTree(seeds)
        self._rng = rng

    def survival(self, distance):
        """Exponential survival function S(d) = exp(-l*d) """
        return np.exp(-distance * self.hazard_rate)

    def acceptance_criterion(self, distance, fraction):
        """If the distance survives then we don't collide yet.
        Cummulative F(d) = 1 - S(d) = Pr(D <= d)

        Args:
            distance (float): The survival probability's argument
            fraction (float): Number in [0, 1] that scales the probability. It is
                useful when different step sizes are used in synthesis for example

        Returns:
            bool: False if we stop, True otherwise
        """
        return not (1.0 - self.survival(distance)) * fraction < self._rng.random()

    def closest_point(self, point):
        """ Closest point on the surface of the convex hull

        Args:
            point (np.ndarray): Point to find its distance to the convex hull vertices

        Returns:
            int: The index of the seed point on the surface of the convex hull
        """
        return self._seed_tree.query(point)

    def __call__(self, point, step_size):
        """ If the point lies inside the convex hull then it doesn't collide.
        If it is outside of it there is an exponential probability of surviving
        that goes down with the distance to the boundary

        Args:
            point (np.ndarray): The query point
            step_size (float): Multiplied with the probability

        Returns:
            bool: True if the collision criterion is satisfied, False otherwise
        """
        if convex_shape_with_point(self.face_points, self.face_normals, point):
            return False

        distance, _ = self.closest_point(point)
        return self.acceptance_criterion(distance, step_size)
