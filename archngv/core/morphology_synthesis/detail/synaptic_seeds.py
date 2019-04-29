import numpy as np

from morphmath import rowwise_dot

from .cell_grid import GridPointRegistry


class PointCloud(object):
    """ 
    """
    def __init__(self, point_array, radius_of_influence, removal_radius):

        self._grid = GridPointRegistry(point_array, radius_of_influence)

        self._size = len(point_array)

        self.radius_of_influence = radius_of_influence
        self.removal_radius = removal_radius

    @property
    def coordinates(self):
        """ Coordinates stored in the point cloud """
        return self._grid.point_array

    def __len__(self):
        """ size """
        return self._size

    def nearest_neighbor(self, point):
        """ Return the nearest neighbor to the given point
        """
        return np.asarray(self._grid.nearest_neighbor(point), dtype=np.intp)

    def ball_query(self, point, radius):
        """ Returns points around sphere with center point and radius """
        points = list(self._grid.ball_query(point, radius))
        return np.asarray(points, dtype=np.float)

    def expanding_ball_query(self, point, number_of_points):
        return np.asarray(list(self._grid.expanding_ball_query(point, number_of_points)), dtype=np.float)

    def average_direction(self, point):
        """ Get average direction from the points around the given point
        which lie inside the incluence_radius
        """
        points = self.ball_query(point, self.radius_of_influence)

        if len(points) == 0:
            return None

        vectors = points - point

        u_dirs = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        average_direction = u_dirs.mean(axis=0)
        return average_direction / np.linalg.norm(average_direction)

    def at_least_n_points_around(self, point, radius, n_points):
        """ Check if there are n_points in the ball located at point with
        radius.
        """
        for n, _ in enumerate(self.ball_query(point, radius)):
            if n == n_points - 1:
                return True
        return False

    def remove(self, point):
        """ Note: mask on points, i.e. on the already sliced array
        """
        self._grid.remove_point(point)
        self._size -= 1

    def remove_points_around(self, point):
        """ Remove the points in the sphere located at point with removal_radius """
        point_list = list(self._grid.ball_query(point, self.removal_radius))
        for point in point_list:
            self.remove(point)
