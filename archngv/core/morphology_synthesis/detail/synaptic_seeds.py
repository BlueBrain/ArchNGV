import numpy as np

from morphmath import rowwise_dot

from .grid_registry import GridPointRegistry

class PointCloud(object):

    def __init__(self, point_array, radius_of_influence, removal_radius):

        self._grid = GridPointRegistry(point_array, radius_of_influence)

        self.__size = len(point_array)

        self.__rr = removal_radius

        self._influence_radius = radius_of_influence

    @property
    def coordinates(self):
        return self._grid.point_array

    def __len__(self):
        return self.__size

    def nearest_neighbor(self, point):
        """ Return the nearest neighbor to the given point
        """
        return np.array(self._grid.nearest_neighbor(point))

    def ball_query(self, point, radius, return_distances=False):
        """ Returns points around point
        """
        coordinates = list(self._grid.ball_query(point, radius))

        if len(coordinates) == 0:
            return coordinates

        coordinates = np.array(coordinates)

        vectors = coordinates - point

        if return_distances:
            dots = rowwise_dot(vectors, vectors)
            return coordinates, vectors, np.sqrt(dots)
        else:
            return coordinates, vectors

    def expanding_ball_query(self, point, number_of_points):
        return np.asarray(list(self._grid.expanding_ball_query(point, number_of_points)), dtype=np.float)

    def average_direction(self, point, radius_of_influence):

        results = self.ball_query(point, radius_of_influence, return_distances=False)

        if len(results) == 0:
            return None

        _, dirs = results

        u_dirs = dirs / np.linalg.norm(dirs, axis=1)[:, np.newaxis]
        average_direction = u_dirs.mean(axis=0)
        return average_direction / np.linalg.norm(average_direction)

    def direction(self, point):
        """ Get average direction from the points around the given point
        which lie inside the incluence_radius
        """
        direction = self.average_direction(point, self._influence_radius)
        return direction

    def remove(self, point):
        """ Note: mask on points, i.e. on the already sliced array
        """
        self._grid.remove_point(point)
        self.__size -= 1

    def at_least_n_points_around(self, point, radius, n_points):

        for n, result in enumerate(self.ball_query(point, radius)):

            if n == n_points - 1:
                return True

        return False

    def remove_points_around(self, point):
        map(self.remove, list(self._grid.ball_query(point, self.__rr)))
