import numpy as np
import collections
import weakref

from morphmath import rowwise_dot
from scipy.spatial import cKDTree


def shell_neighborhood(level):

    n_range = range(-level, level + 1)

    for i in (-level, level):
        for j in n_range:
            for k in n_range:
                yield (i, j, k)

    r_range = range(-level + 1, level)

    for j in (-level, level):
        for i in r_range:
            for k in n_range:
                yield (i, j, k)

    for k in (-level, level):
        for i in r_range:
            for j in r_range:
                yield (i, j, k)


def calculate_cell_neighbors(Lx, Ly):
    return (

     -Ly - Lx - 1,
     -Lx - 1,
     Lx * Ly - Lx - 1,
     - Lx * Ly - 1,
     - 1,
     Lx * Ly - 1,
     - Lx * Ly + Lx - 1,
     Lx - 1,
     Lx * Ly + Lx - 1,
     - Lx * Ly - Lx,
     - Lx,
     Lx * Ly - Lx,
     - Lx * Ly,
     Lx * Ly,
     - Lx * Ly + Lx,
     Lx,
     Lx * Ly + Lx,
     -Lx*Ly - Lx + 1,
     -Lx + 1,
     Lx*Ly - Lx + 1,
     - Lx * Ly + 1,
     1,
     Lx * Ly + 1,
     -Lx*Ly + Lx + 1,
     Lx + 1,
     Lx * Ly + Lx + 1

    )


class Point(object):
    __slots__=['x', 'y', 'z', 'neighbors', '__weakref__']
    def __init__(self, point):

        self.x = point[0]
        self.y = point[1]
        self.z = point[2]

        self.neighbors = weakref.WeakSet()

    def __getitem__(self, index):
        return (self.x, self.y, self.z)[index]


    def __hash__(self):
        return hash((round(self.x, 2),
                round(self.y, 2),
                round(self.z, 2)))

    def __eq__(self, othr):
        return abs(self.x - othr.x) < 1e-2 and \
               abs(self.y - othr.y) < 1e-2 and \
               abs(self.z - othr.z) < 1e-2

    def __str__(self):
        return "Point({}, {}, {})".format(self.x, self.y, self.z)

    __repr__ = __str__



class GridPointRegistry(collections.MutableMapping):

    def __init__(self, point_array, cutoff_distance):

        self._ex = np.ptp(point_array[:, 0])
        self._ey = np.ptp(point_array[:, 1])
        self._ez = np.ptp(point_array[:, 2])

        self._offx = np.min(point_array[:, 0])
        self._offy = np.min(point_array[:, 1])
        self._offz = np.min(point_array[:, 2])

        self._rc2 = cutoff_distance ** 2
        self._dl = cutoff_distance / np.sqrt(2)
        self._inv_dl = 1. / self._dl

        self._sx = int(np.ceil(self._ex * self._inv_dl))
        self._sy = int(np.ceil(self._ey * self._inv_dl))
        self._sz = int(np.ceil(self._ez * self._inv_dl))

        self._NMAX = self._sx * self._sy * self._sz

        self._sxsy = self._sx * self._sy

        points = map(Point, point_array)
        indices = map(self.point_to_index, points)
        self.store = {index: set() for index in indices}
        map(lambda i: self.store[indices[i]].add(points[i]), xrange(len(points)))

    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return self.point_to_index(key) if isinstance(key, Point) else key

    def point_to_index(self, point):
        i = int((point.x - self._offx) * self._inv_dl)
        j = int((point.y - self._offy) * self._inv_dl)
        k = int((point.z - self._offz) * self._inv_dl)
        return  self.ijk_to_index(i, j, k)

    def ijk_to_index(self, i, j, k):
        return i + self._sx * j + self._sxsx * k


    def ball_query(self, point, radius):

        r2 = radius ** 2

        p_object_i = Point(point)

        index = self.cell_registry.point_to_index(p_object_i)

        if index in self.cell_registry:
            for p_object_j in self.cell_registry[index]:
                if self._squared_distance(point_object_i, point_object_j) <= r2:
                    yield (p_object_j.x, p_object_j.y, p_object_j.z)

        for i, j, k in shell_neighborhood(1):
            new_index = index + self.cell_registry.ijk_to_index(i, j, k)
            if self._is_index_valid(new_index):
                for p_object_j in self.cell_registry[new_index]:
                    if self._squared_distance(point_object_i, point_object_j) <= r2:
                        yield (p_object_j.x, p_object_j.y, p_object_j.z)




    def nearest_neighbor(self, point, return_distance=False):

        p_object_i = Point(point)

        index = self.cell_registry.point_to_index(p_object_i)


        closest_distance = np.inf
        closest_point = None


        if index in self.cell_registry:

            for p_object_j in self.cell_registry[index]:

                d2 = self._squared_distance(p_object_i, p_object_j)

                if d2 < closest_distance:
                    closest_distance = d2
                    closest_point = p_object_j

        else:

            visited = set([index])

            level = 1

            while 1:

                for i, j, k in shell_neighborhood(level):

                    new_index = \
                    index + self.cell_registry.ijk_to_index(i, j, k)

                    if new_index not in visited and \
                       self._is_index_valid(new_index):

                        for p_object_j in self.cell_registry[new_index]:

                            d2 = self._squared_distance(p_object_i, p_object_j)

                            if d2 < closest_distance:
                                closest_distance = d2
                                closest_point = p_object_j

                        visited.add(new_index)

                if closest_point is not None:
                    break

                level += 1


        if return_distance:
            return (closest_point.x, closest_point.y, closest_point.z), np.srt(closest_distance)
        else:
            return (closest_point.x, closest_point.y, closest_point.z)

    def _is_index_valid(self, index):
        return index in self.store and 0 <= index < self._NMAX

    def remove_point(self, point):

        point_object = Point(point)

        point_set = self.cell_registry[point_object]
        point_set.remove(point_object)

        # if set is empty, delete the cell altogether
        if not point_set:
            del self.cell_registry[point_object]



class GridNeighbors(object):

    def __init__(self, point_array, cutoff_distance):

        self._rc = cutoff_distance

        self.cell_registry = GridPointRegistry(point_array, cutoff_distance)

        self._neighbor_offsets = \
        calculate_cell_neighbors(self.cell_registry._sx, self.cell_registry._sy)

        points = self.point_objects
        arr_points = self.point_array
        map(lambda t: points[t[0]].neighbors.update([points[index] for index in t[1] if index != t[0]]),
            enumerate(cKDTree(arr_points, copy_data=False).query_ball_point(arr_points, self._rc, eps=1e-2, n_jobs=8)))

    @property
    def point_objects(self):
        return tuple(p for cell in self.cell_registry.values() for p in cell)

    @property
    def point_array(self):
        return np.array([(p.x ,p.y, p.z) for p in self.point_objects])

    def _squared_distance(p0, p1):
        return (p0.x - p1.x) ** 2 + (p0.y - p1.y) ** 2 + (p0.z - p1.z) ** 2

    def _neighboring_cell_point_sets(self, cell_index):

        nmax = self.cell_registry._NMAX

        idx = (cell_index + value for value in self._neighbor_offsets)
        return [self.cell_registry[index] for index in idx if 0 < index < nmax and index in self.cell_registry]




        

