""" Hashmap experimental stuff
"""


class HashMapBase:
    """Base of hashmap"""

    def __init__(self, voxel_size, offset):

        self._factor = 1.0 / float(voxel_size)
        self.offx, self.offy, self.offz = offset
        self._d = {}

    def key(self, point):
        """ijk key from 3D point"""
        return (
            int((point[0] - self.offx) * self._factor),
            int((point[1] - self.offy) * self._factor),
            int((point[2] - self.offz) * self._factor),
        )

    @property
    def factor(self):
        """Returns factor"""
        return self._factor


class PointHashMap(HashMapBase):
    """Hashmap for points"""

    def add_point(self, point):
        """Add a new point to he map"""
        key = self.key(point)
        try:
            self._d[key].append(point)
        except KeyError:
            self._d[key] = [point]

    def _gen(self, ijk_min, ijk_max):
        """Yields the points that are found in the interval
        [ijk_min, ijk_max]
        """
        i_min, j_min, k_min = ijk_min
        i_max, j_max, k_max = ijk_max

        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                for k in range(k_min, k_max + 1):
                    if (i, j, k) in self._d:
                        for point in self._d[(i, j, k)]:
                            yield point

    def q_window(self, xmin, ymin, zmin, xmax, ymax, zmax):
        """Returns the points contained in the bounding rectangle"""
        return list(self._gen(self.key((xmin, ymin, zmin)), self.key((xmax, ymax, zmax))))
