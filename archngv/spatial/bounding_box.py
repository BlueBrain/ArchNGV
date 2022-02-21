""" Bounding box related functions
"""
# pylint: disable = too-many-locals

import numpy as np

from .inclusion import points_in_rectangle, spheres_in_rectangle


class BoundingBox:
    """Bounding box data object"""

    @classmethod
    def from_points(cls, points):
        """bbox constructor from point cloud"""
        min_coordinates = points.min(axis=0)
        max_coordinates = points.max(axis=0)
        return cls(min_coordinates, max_coordinates)

    @classmethod
    def from_spheres(cls, points, radii):
        """bbox constructor from spheres"""
        min_coordinates = (points - radii[:, np.newaxis]).min(axis=0)
        max_coordinates = (points + radii[:, np.newaxis]).max(axis=0)
        return cls(min_coordinates, max_coordinates)

    def __init__(self, min_coordinates, max_coordinates):
        self._bb = np.array((min_coordinates, max_coordinates), dtype=np.float32)

    def __eq__(self, other):
        """Equality of bboxes"""
        return np.allclose(self.min_point, other.min_point) and np.allclose(
            self.max_point, other.max_point
        )

    def __add__(self, other):
        """Create a new bbox that is the support of both bboxes"""
        return BoundingBox(
            np.min((self.min_point, other.min_point), axis=0),
            np.max((self.max_point, other.max_point), axis=0),
        )

    def __str__(self):
        return "BoundingBox(min_point={}, max_point={})".format(self.min_point, self.max_point)

    __repr__ = __str__

    @property
    def ranges(self):
        """x_range, y_range, z_range"""
        return self._bb

    @property
    def offset(self):
        """Get the offset from the origin"""
        return self._bb[0]

    @property
    def min_point(self):
        """Return the min point"""
        return self.offset

    @property
    def max_point(self):
        """Return the max point"""
        return self._bb[1]

    @property
    def center(self):
        """Center of bounding box"""
        return 0.5 * (self.min_point + self.max_point)

    @property
    def extent(self):
        """Get the difference between the max and min range per dimension"""
        return np.array(
            (
                (self._bb[1][0] - self._bb[0][0]),
                (self._bb[1][1] - self._bb[0][1]),
                (self._bb[1][2] - self._bb[0][2]),
            )
        )

    @property
    def layout(self):
        """Get the points and the respective edges of the bounding box"""
        (xmin, ymin, zmin, xmax, ymax, zmax) = self._bb.ravel()

        points = np.array(
            (
                (xmin, ymin, zmin),
                (xmin, ymax, zmin),
                (xmin, ymin, zmax),
                (xmin, ymax, zmax),
                (xmax, ymin, zmin),
                (xmax, ymax, zmin),
                (xmax, ymin, zmax),
                (xmax, ymax, zmax),
            )
        )

        edges = np.array(
            (
                (0, 1),
                (0, 2),
                (0, 4),
                (1, 3),
                (1, 5),
                (2, 3),
                (2, 6),
                (3, 7),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 7),
            )
        )

        return points, edges

    @property
    def volume(self):
        """Volume of bbox"""
        return np.abs(np.prod(self.extent))

    def translate(self, point):
        """Translate by point coordinates"""
        self._bb += point

    def scale(self, triplet):
        """Scale bbox"""
        center = self.center
        self.translate(-center)
        self._bb *= triplet
        self.translate(center)

    def points_inside(self, points):
        """Returns a boolean mask of the points included in the
        bounding box
        """
        return points_in_rectangle(points, self.min_point, self.max_point)

    def spheres_inside(self, centers, radii):
        """Returns a boolean mask of the spheres that are included in the
        bounding box
        """
        return spheres_in_rectangle(centers, radii, self.min_point, self.max_point)


def aabb_point(point):
    """Returns the bounding box of a point"""
    return (point[0], point[1], point[2], point[0], point[1], point[2])


def aabb_point_cloud(points):
    """Returns the bounding box of a collection of points"""
    min_coos = points.min(axis=0)
    max_coos = points.max(axis=0)

    return (
        min_coos[0],
        min_coos[1],
        min_coos[2],
        max_coos[0],
        max_coos[1],
        max_coos[2],
    )


def aabb_sphere(center, radius):
    """Returns the bounding box of a sphere given its center and radius.
    (xmin, ymin, zmin, xmax, ymax, zmax)
    """
    return (
        center[0] - radius,
        center[1] - radius,
        center[2] - radius,
        center[0] + radius,
        center[1] + radius,
        center[2] + radius,
    )


def aabb_disk(center, radius, normal):
    """Returns the bounding box of a disk"""
    e_v = radius * np.sqrt(1.0 - normal**2)

    return (
        center[0] - e_v[0],
        center[1] - e_v[1],
        center[2] - e_v[2],
        center[0] + e_v[0],
        center[1] + e_v[1],
        center[2] + e_v[2],
    )


def aabb_tapered_capsule(cap1_center, cap2_center, cap1_radius, cap2_radius):
    """Returns the bbox of a tapered capsule"""
    xmin1, ymin1, zmin1, xmax1, ymax1, zmax1 = aabb_sphere(cap1_center, cap1_radius)
    xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = aabb_sphere(cap2_center, cap2_radius)

    return (
        min(xmin1, xmin2),
        min(ymin1, ymin2),
        min(zmin1, zmin2),
        max(xmax1, xmax2),
        max(ymax1, ymax2),
        max(zmax1, zmax2),
    )


def aabb_capsule(cap1_center, cap2_center, height):
    """Returns the bounding box of a capsule"""
    return aabb_tapered_capsule(cap1_center, cap2_center, height, height)


def aabb_cylinder(cap_center1, cap_center2, radius):
    """Returns the bounding box of a cylinder"""
    u_v = cap_center2 - cap_center1

    e_v = radius * np.sqrt(1.0 - u_v**2 / np.dot(u_v, u_v))

    return (
        min(cap_center1[0] - e_v[0], cap_center2[0] - e_v[0]),
        min(cap_center1[1] - e_v[1], cap_center2[1] - e_v[1]),
        min(cap_center1[2] - e_v[2], cap_center2[2] - e_v[2]),
        max(cap_center1[0] + e_v[0], cap_center2[0] + e_v[0]),
        max(cap_center1[1] + e_v[1], cap_center2[1] + e_v[1]),
        max(cap_center1[2] + e_v[2], cap_center2[2] + e_v[2]),
    )
