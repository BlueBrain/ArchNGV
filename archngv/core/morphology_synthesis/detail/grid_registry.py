import collections
import numpy as np

def _squared_distance(p0, p1):
    return (p0.x - p1.x) ** 2 + (p0.y - p1.y) ** 2 + (p0.z - p1.z) ** 2

def _shell_neighborhood(level):
    """ Returns the ijk indices that correspond to 
    a cubic shell of specific level. For example level 1
    will return the ijk for the 26 grid neighbors from the
    grid origin, level 2 the next cell of 98 neighbors etc.
    It does not include the ijk of the previous level (I.e. it's a shell).
 
               2
      1    . . . . . 
    . . .  .       .
    .   .  .       .
    . . .  .       .
           . . . . .
           
    """

    if level == 0:

        yield (0, 0, 0)

    else:

        n_range = list(range(-level, level + 1))

        for i in (-level, level):
            for j in n_range:
                for k in n_range:
                    yield (i, j, k)


        r_range = list(range(-level + 1, level))

        for j in (-level, level):
            for i in r_range:
                for k in n_range:
                    yield (i, j, k)

        for k in (-level, level):
            for i in r_range:
                for j in r_range:
                    yield (i, j, k)


class Point(object):
    __slots__=['x', 'y', 'z', 'neighbors', '__weakref__']
    def __init__(self, point):

        self.x = point[0]
        self.y = point[1]
        self.z = point[2]

        self.neighbors = None

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

        ex, ey, ez = np.ptp(point_array, axis=0)
        self._offx, self._offy, self._offz = np.min(point_array, axis=0)

        self._rc2 = cutoff_distance ** 2

        self._dl = 2. * cutoff_distance / np.sqrt(2)
        self._inv_dl = 1. / self._dl

        self._sx = int(np.ceil(ex * self._inv_dl))
        self._sy = int(np.ceil(ey * self._inv_dl))
        self._sz = int(np.ceil(ez * self._inv_dl))
        self._sxsy = self._sx * self._sy

        self._NMAX = self._sx * self._sy * self._sz

        indices = list(map(self.point_to_index, points))

        self.store = {index: set() for index in indices}

        points = map(Point, point_array)
        for (i, point) in enumerate(points):
            self.store[indices[i]].add(point)

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

    def __keytransform__(self, index):
        return index

    @property
    def point_objects(self):
        return tuple(p for cell in self.store.values() for p in cell)

    @property
    def point_array(self):
        return np.array([(p.x ,p.y, p.z) for p in self.point_objects])

    def point_to_ijk(self, point):
        i = int((point.x - self._offx) * self._inv_dl)
        j = int((point.y - self._offy) * self._inv_dl)
        k = int((point.z - self._offz) * self._inv_dl)
        return (i, j, k)

    def ijk_to_index(self, i, j, k):
        return i + self._sx * j + self._sxsy * k

    def point_to_index(self, point):
        return  self.ijk_to_index(*self.point_to_ijk(point))

    def is_index_valid(self, index):
        return index in self.store and 0 <= index < self._NMAX

    def upper_level_from_radius(self, radius):
        return int(np.ceil(2. * radius) / self._dl)

    def ball_query(self, point, radius):

        r2 = radius ** 2

        p_object_i = Point(point)
        current_index = self.point_to_index(p_object_i)

        visited = set()

        for level in range(self.upper_level_from_radius(radius) + 1):
            for i, j, k in _shell_neighborhood(level):
                new_index = current_index + self.ijk_to_index(i, j, k)
                if new_index not in visited and self.is_index_valid(new_index):
                    for p_object_j in self.store[new_index]:
                        if _squared_distance(p_object_i, p_object_j) <= r2:
                            yield (p_object_j.x, p_object_j.y, p_object_j.z)
                    visited.add(new_index)

    def expanding_ball_query(self, point, number_of_points):

        p_object_i = Point(point)
        current_index = self.point_to_index(p_object_i)


        visited = set()
        level = 0
        n = 0

        while n < number_of_points:
            for i, j, k in _shell_neighborhood(level):
                new_index = current_index + self.ijk_to_index(i, j, k)
                if new_index not in visited and self.is_index_valid(new_index):
                    for p_object_j in self.store[new_index]:
                        yield (p_object_j.x, p_object_j.y, p_object_j.z)
                        n += 1
                    visited.add(new_index)
            level += 1




    def nearest_neighbor(self, point, return_distance=False):

        p_object_i = Point(point)

        index = self.point_to_index(p_object_i)

        closest_distance = np.inf
        closest_point = None

        if index in self.store:

            for p_object_j in self.store[index]:

                d2 = _squared_distance(p_object_i, p_object_j)

                if d2 < closest_distance:
                    closest_distance = d2
                    closest_point = p_object_j

        else:

            visited = set([index])

            level = 1

            while 1:

                for i, j, k in _shell_neighborhood(level):

                    new_index = \
                    index + self.ijk_to_index(i, j, k)

                    if new_index not in visited and \
                       self.is_index_valid(new_index):

                        for p_object_j in self.store[new_index]:

                            d2 = _squared_distance(p_object_i, p_object_j)

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

    def remove_point(self, point):

        point_object = Point(point)

        index = self.point_to_index(point_object)

        point_set = self.store[index]
        point_set.remove(point_object)

        # if set is empty, delete the cell altogether
        if not point_set:
            del self.store[index]
