
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import Delaunay


class Boundary(object):

    def __init__(self, points, faces_vertices):

        self._points = points
        self._faces_vertices = faces_vertices

    @property
    def points(self):
        return self._points

    @property
    def faces(self):
        return self.points[self._face_vertices]

    @property
    def face_normals(self):
        faces = self.faces

        # vectors for two consecutive sides
        v1 = faces[:, 1, :] - faces[:, 0, :]
        v2 = faces[:, 2, :] - faces[:, 1, :]
        return np.cross(v1, v2)


class ConvexBoundary(Boundary):

    def __init__(self, points, faces_vertices):
        super(ConvexBoundary, self).__init__(points, faces_vertices)

    @property
    def centroid(self):
        return np.mean(self.vertices, axis=0)

    def inclusion(self, points):
        return Delaunay(self.points).find_simplex(points) >= 0

