""" Shapes Data Structures
"""
from typing import Set, Tuple

import numpy
from cached_property import cached_property
from scipy.spatial import ConvexHull  # pylint: disable = no-name-in-module

from archngv.utils import ngons
from archngv.utils.linear_algebra import rowwise_dot

from . import bounding_box as _bbox
from . import support_functions as _sup
from . import utils as _ut


class Sphere:
    """Sphere class object."""

    __slots__ = ["center", "radius"]

    def __init__(self, center, radius):

        self.center = center
        self.radius = radius

    @property
    def centroid(self):
        """Returns the centroid / center of sphere"""
        return self.center

    @property
    def bounding_box(self):
        """Returns the bounding box of the sphere"""
        return _bbox.aabb_sphere(self.center, self.radius)

    def support(self, unit_direction):
        """Returns the support of the sphere."""
        return _sup.sphere(self.center, self.radius, unit_direction)


class ConvexPolygon:
    """Convex polygon data structure"""

    @classmethod
    def from_point_cloud(cls, points: numpy.ndarray) -> "ConvexPolygon":
        """Constructor from point cloud"""
        return cls(points, ConvexHull(points).simplices)

    @classmethod
    def from_convex_hull(cls, convex_hull: ConvexHull) -> "ConvexPolygon":
        """Constructor from scipy convex_hull"""
        return cls(convex_hull.points, convex_hull.simplices)

    def __init__(self, points: numpy.ndarray, triangles: numpy.ndarray):
        self._points = points
        self._triangles = triangles

    @property
    def points(self) -> numpy.ndarray:
        """Returns convex polygon points"""
        return self._points

    @points.setter
    def points(self, new_points: numpy.ndarray):
        """Update polygon points"""
        self._points = new_points

    @cached_property
    def triangles(self) -> numpy.ndarray:
        """Returns convex polygon triangles. Flips the sequence of the triangles
        if the normal is inwards ensure that every time normals are calculated,
        the normals are outward
        """
        return _ut.make_normals_outward(self.centroid, self.points, self._triangles)

    @property
    def face_vectors(self) -> numpy.ndarray:
        """Sequential face vectors"""
        ps, tris = self.points, self.triangles
        return ngons.vectorized_consecutive_triangle_vectors(ps, tris)

    @property
    def face_points(self) -> numpy.ndarray:
        """Returns one point for each face."""
        return self.points[self.triangles[:, 0]]

    @property
    def face_areas(self) -> numpy.ndarray:
        """Returns areas of faces"""
        return ngons.vectorized_triangle_area(*self.face_vectors)

    @property
    def face_normals(self) -> numpy.ndarray:
        """Returns normals of faces"""
        return ngons.vectorized_triangle_normal(*self.face_vectors)

    @property
    def face_centers(self) -> numpy.ndarray:
        """Returns centers of faces"""
        return self.points[self.triangles].mean(axis=1)

    @cached_property
    def centroid(self) -> numpy.ndarray:
        """Arithmetic mean of the vertices"""
        return numpy.mean(self.points, axis=0)

    @property
    def volume(self) -> float:
        """Volume of the convex polygon"""
        vecs = self.points[self.triangles] - self.centroid
        return ngons.vectorized_tetrahedron_volume(
            vecs[:, 0, :], vecs[:, 1, :], vecs[:, 2, :]
        ).sum()

    @property
    def bounding_box(self) -> Tuple[float, ...]:
        """Axis aligned bounding box of convex polygon"""
        return _bbox.aabb_point_cloud(self.points)

    def support(self, unit_direction) -> numpy.ndarray:
        """Support of convex polygon along unit direction"""
        return _sup.convex_polytope(self.points, self.adjacency, unit_direction)

    @property
    def adjacency(self) -> Tuple[Set[int], ...]:
        """Adjacency matrix of its vertices"""
        adjacency: Tuple[Set[int], ...] = tuple(set() for _ in range(len(self.points)))
        for vertices in self.triangles:
            for i, vertex in enumerate(vertices):
                try:
                    for j in vertices[(i + 1) :]:
                        adjacency[vertex].add(j)
                        adjacency[j].add(vertex)
                except IndexError:
                    pass
        return adjacency

    @property
    def inscribed_sphere(self) -> Tuple[numpy.ndarray, float]:
        """Returns centroid and radius of a sphere that is inscribed
        inside the convex polygon
        """
        centroid = self.centroid
        face_first_points = self.points[self.triangles[:, 0]]
        radius = min(rowwise_dot(self.face_normals, face_first_points - centroid))
        return centroid, radius


class TaperedCapsule:
    """Capsule data structure"""

    __slots__ = ["cap1_center", "cap2_center", "cap1_radius", "cap2_radius"]

    def __init__(self, cap1_center, cap2_center, cap1_radius, cap2_radius):

        self.cap1_center = cap1_center
        self.cap2_center = cap2_center
        self.cap1_radius = cap1_radius
        self.cap2_radius = cap2_radius

    @property
    def centroid(self):
        """Centroid of capsule"""
        raise NotImplementedError

    @property
    def bounding_box(self):
        """aabb capsule"""
        return _bbox.aabb_tapered_capsule(
            self.cap1_center, self.cap2_center, self.cap1_radius, self.cap2_radius
        )

    def support(self, normalized_direction):
        """Support of capsule"""
        raise NotImplementedError


class Cylinder:
    """Cylinder data structure"""

    __slots__ = ["cap1_center", "cap2_center", "radius"]

    def __init__(self, cap1_center, cap2_center, radius):

        self.cap1_center = cap1_center
        self.cap2_center = cap2_center
        self.radius = radius

    @property
    def centroid(self):
        """Geometrical centroid"""
        return 0.5 * (self.cap1_center + self.cap2_center)

    @property
    def bounding_box(self):
        """aabb"""
        return _bbox.aabb_cylinder(self.cap1_center, self.cap2_center, self.radius)

    @property
    def central_axis(self):
        """Vector connecting its two cap centers"""
        return self.cap2_center - self.cap1_center

    @property
    def height(self):
        """Length of central axis"""
        return numpy.linalg.norm(self.central_axis)

    def support(self, unit_direction):
        """Support of cylinder"""
        return _sup.cylinder(
            self.centroid, self.central_axis, self.radius, self.height, unit_direction
        )
