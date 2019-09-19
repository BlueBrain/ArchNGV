""" Shapes Data Structures
"""
# pylint: disable = useless-object-inheritance


import numpy
from scipy.spatial import ConvexHull  # pylint: disable = no-name-in-module

import archngv.math_utils as mt

from . import bounding_box as _bbox
from . import support_functions as _sup


from . import utils as _ut


class Sphere(object):
    """ Sphere class object.
    """
    __slots__ = ['center', 'radius']

    def __init__(self, center, radius):

        self.center = center
        self.radius = radius

    @property
    def centroid(self):
        """ Returns the centroid / center of sphere
        """
        return self.center

    @property
    def bounding_box(self):
        """ Returns the bounding box of the sphere
        """
        return _bbox.aabb_sphere(self.center, self.radius)

    def support(self, unit_direction):
        """ Returns the support of the sphere.
        """
        return _sup.sphere(self.center, self.radius, unit_direction)


class ConvexPolygon(object):
    """ Convex polygon data structure
    """
    __slots__ = 'points', 'triangles'

    @classmethod
    def from_point_cloud(cls, points):
        """ Constructor from point cloud
        """
        return cls(points, ConvexHull(points).simplices)

    @classmethod
    def from_convex_hull(cls, convex_hull):
        """ Constructor from scipy convex_hull
        """
        return cls(convex_hull.points, convex_hull.simplices)

    def __init__(self, points, face_vertices):

        self.points = points

        # break down polygonal faces to triangular ones
        gen = _ut.triangles_from_polygons_generator(points, face_vertices)

        self.triangles = _ut.fromiter2D(gen, 3, numpy.intp)

        # flip the sequence of the triangles if the normal is inwards
        # ensure that everytime normals are calculated, the normals are outward
        self.triangles = _ut.make_normals_outward(self._center,
                                                  self.points,
                                                  self.triangles,
                                                  self.face_normals)

    @property
    def face_vectors(self):
        """ Sequential face vectors
        """
        tris = self.triangles
        pnts = self.points
        return (pnts[tris[:, 1]] - pnts[tris[:, 0]],
                pnts[tris[:, 2]] - pnts[tris[:, 0]])

    @property
    def face_points(self):
        """ Returns one point for each face. """
        return self.points[self.triangles[:, 0]]

    @property
    def face_areas(self):
        """ Returns areas of faces
        """
        return mt.vectorized_triangle_area(*self.face_vectors)

    @property
    def face_normals(self):
        """ Returns normals of faces """
        return mt.vectorized_triangle_normal(*self.face_vectors)

    @property
    def face_centers(self):
        """ Returns centers of faces """
        return self.points[self.triangles].mean(axis=1)

    @property
    def _center(self):
        """ Arithmetic mean of the vertices """
        return numpy.mean(self.points, axis=0)

    @property
    def centroid(self):
        """ Calculation of the center of mass of the polyhedron by taking the arithmetic
        center inside and decomposing the polyhedron into tetrahedra formed by that center
        and each triangular face.
        """
        tetrahedra_centroids = (self.points[self.triangles].sum(axis=1) + self._center) / 4.

        vecs = self.points[self.triangles] - self._center
        volumes = \
            mt.vectorized_tetrahedron_volume(vecs[:, 0, :], vecs[:, 1, :], vecs[:, 2, :])

        return numpy.average(tetrahedra_centroids,
                             weights=volumes / volumes.sum(), axis=0)

    @property
    def volume(self):
        """ Volume of the convex polygon
        """
        vecs = self.points[self.triangles] - self.centroid
        return mt.vectorized_tetrahedron_volume(vecs[:, 0, :],
                                                vecs[:, 1, :],
                                                vecs[:, 2, :]).sum()

    @property
    def bounding_box(self):
        """ Axis aligned bounding box of convex polygon
        """
        return _bbox.aabb_point_cloud(self.points)

    def support(self, unit_direction):
        """ Support of convex polygon along unit direction
        """
        return _sup.convex_polytope(self.points, self.adjacency, unit_direction)

    @property
    def adjacency(self):
        """ Adjacency matrix of its vertices
        """
        adjacency = tuple(set() for _ in range(len(self.points)))
        for vertices in self.triangles:
            for i, vertex in enumerate(vertices):
                try:
                    for j in vertices[(i + 1):]:
                        adjacency[vertex].add(j)
                        adjacency[j].add(vertex)
                except IndexError:
                    pass
        return adjacency

    @property
    def inscribed_sphere(self):
        """ Returns centroid and radius of a sphere that is inscribed
        inside the convex polygon
        """

        centroid = self.centroid

        face_first_points = self.points[self.triangles[:, 0]]

        radius = min(mt.rowwise_dot(self.face_normals, face_first_points - centroid))

        return centroid, radius

    def scale(self, scale_factor, inplace=False):
        """ Uniformly scales the polygon by a scale_factor, assuming its centroid
        sits on the origin.
        """
        cnt = self.centroid

        if inplace:
            self.points = scale_factor * (self.points - cnt) + cnt
            return self

        return ConvexPolygon(scale_factor * (self.points - cnt) + cnt, self.triangles)


class TaperedCapsule(object):
    """ Capsule data structure
    """
    __slots__ = ['cap1_center', 'cap2_center', 'cap1_radius', 'cap2_radius']

    def __init__(self, cap1_center, cap2_center, cap1_radius, cap2_radius):

        self.cap1_center = cap1_center
        self.cap2_center = cap2_center
        self.cap1_radius = cap1_radius
        self.cap2_radius = cap2_radius

    @property
    def centroid(self):
        """ Centroid of capsule """
        raise NotImplementedError

    @property
    def bounding_box(self):
        """ aabb capsule """
        return _bbox.aabb_tapered_capsule(self.cap1_center,
                                          self.cap2_center,
                                          self.cap1_radius,
                                          self.cap2_radius)

    def support(self, normalized_direction):
        """ Support of capsule """
        raise NotImplementedError


class Cylinder(object):
    """ Cylinder data structure
    """
    __slots__ = ['cap1_center', 'cap2_center', 'radius']

    def __init__(self, cap1_center, cap2_center, radius):

        self.cap1_center = cap1_center
        self.cap2_center = cap2_center
        self.radius = radius

    @property
    def centroid(self):
        """ Geometrical centroid """
        return 0.5 * (self.cap1_center + self.cap2_center)

    @property
    def bounding_box(self):
        """ aabb """
        return _bbox.aabb_cylinder(self.cap1_center,
                                   self.cap2_center,
                                   self.radius)

    @property
    def central_axis(self):
        """ Vector connecting its two cap centers
        """
        return self.cap2_center - self.cap1_center

    @property
    def height(self):
        """ Length of central axis """
        return numpy.linalg.norm(self.central_axis)

    def support(self, unit_direction):
        """ Support of cylinder
        """
        return _sup.cylinder(self.centroid,
                             self.central_axis,
                             self.radius,
                             self.height, unit_direction)
