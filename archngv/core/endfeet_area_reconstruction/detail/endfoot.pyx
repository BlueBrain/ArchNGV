#cython: auto_pickle=False

cimport cython

import numpy as np
cimport numpy as np

from morphmath import vectorized_triangle_area

from libc.math cimport sqrt

import logging

ctypedef np.npy_uintp SIZE_t


L = logging.getLogger(__name__)



def _calculate_neighbors(triangles):
        ns = {}
        for (v1, v2, v3) in triangles:
            for key, neighs in [(v1, (v2, v3)), (v2, (v3, v1)), (v3, (v1, v2))]:
                try:
                    ns[key].update(neighs)
                except KeyError:
                    ns[key] = set(neighs)
        return ns

def _vertex_to_triangles(triangles):

    v2t = {}

    for index, triangle in enumerate(triangles):
        for vertex in triangle:
            try:
                v2t[vertex].add(index)
            except KeyError:
                v2t[vertex] = set([index])
    return v2t


def _edge_to_triangles(triangles):

    et = {}

    # Contour edges belong to only one triangle
    for (index, (v1, v2, v3)) in  enumerate(triangles):
        for (c1, c2) in ((v1, v2), (v2, v3), (v3, v1)):

            e = frozenset((c1, c2))
            if e in et:
                et[e].add(index)
            else:
                et[e] = set([index])
    return et


def _remap_triangle_indices(triangles, return_indices=False):

    unique_indices = \
    sorted(set(vertex for triangle in new_triangles for vertex in triangle))

    m = {old: new for new, old in enumerate(unique_indices)}

    new_triangles = np.asarray([(m[tr[0]], m[tr[1]], m[tr[2]]) for tr in triangles], dtype=np.uintp)

    if return_indices:

        return new_triangles, np.asarray(unique_indices, dtype=np.uintp)

    else:

        return new_triangles


def _slice_coordinates_and_remap_indices(coordinates, triangles, return_indices=False):

    new_triangles, unique_indices = _remap_triangle_indices(triangles)

    if return_indices:

        return coordinates[unique_indices], new_triangles, unique_indices

    else:

        return coordinates[unique_indices], new_triangles


cdef tuple subset_triangles_that_include_vertices(SIZE_t[:, :] triangles, set indices):

    cdef SIZE_t n, k
    cdef SIZE_t[:, :] new_triangles = np.empty((len(triangles), 3), dtype=np.uintp)

    k = 0
    for n in range(len(triangles)):

        if triangles[n, 0] in indices and \
           triangles[n, 1] in indices and \
           triangles[n, 2] in indices:

           new_triangles[k] = triangles[n]

           k += 1

    return new_triangles[:k], set(v for tr in new_triangles[:k] for v in tr)

cdef tuple subset_triangles_that_do_not_include_vertices(SIZE_t[:, :] triangles,
                                                         set indices):

    cdef SIZE_t n, k
    cdef SIZE_t[:, :] new_triangles = np.empty((len(triangles), 3), dtype=np.uintp)

    k = 0
    for n in range(len(triangles)):

        if triangles[n, 0] not in indices and \
           triangles[n, 1] not in indices and \
           triangles[n, 2] not in indices:

           new_triangles[k] = triangles[n]

           k += 1

    return new_triangles[:k], set(v for tr in new_triangles[:k] for v in tr)



cdef inline float triangle_area(float ax, float ay, float az,
                                float bx, float by, float bz,
                                float cx, float cy, float cz) nogil:

    cdef:
        float u1 = bx - ax
        float u2 = by - ay
        float u3 = bz - az

        float v1 = cx - ax
        float v2 = cy - ay
        float v3 = cz - az

    return 0.5 * sqrt((u2 * v3 - u3 * v2) ** 2 + \
                      (u1 * v3 - u3 * v1) ** 2 + \
                      (u1 * v2 - u2 * v1) ** 2)


cpdef object create_endfoot_from_global_data(SIZE_t index,
                                             float[:, :] all_coordinates,
                                             SIZE_t[:, :] all_triangles,
                                             SIZE_t[:] endfoot_indices):

        cdef:

            SIZE_t[:, :] local_triangles

            set vasculature_set_indices

            list sorted_indices

            SIZE_t n

            float[:, :] local_coordinates

        local_triangles, vasculature_set_indices = \
        subset_triangles_that_include_vertices(all_triangles, set(endfoot_indices))

        sorted_indices = sorted(vasculature_set_indices)

        local_coordinates = np.empty((len(sorted_indices), 3), dtype=np.float32)

        # slicing the coordinates we jump to the local reference system
        for n in range(len(sorted_indices)):
            local_coordinates[n] = all_coordinates[sorted_indices[n]]

        # therefore the triangle indices have to be remapped acccordingly
        global_to_local_map = {old: new for new, old in enumerate(sorted_indices)}


        for n in range(len(local_triangles)):
            local_triangles[n, 0] = global_to_local_map[local_triangles[n, 0]]
            local_triangles[n, 1] = global_to_local_map[local_triangles[n, 1]]
            local_triangles[n, 2] = global_to_local_map[local_triangles[n, 2]]

        # finally we need to keep a record of the mapping between the local and global
        # reference system in order to be able to go back later
        local_to_global_map = {new: old for new, old in enumerate(sorted_indices)}

        return Endfoot(index, local_coordinates, local_triangles, set(vasculature_set_indices), local_to_global_map)


cdef class Endfoot:

    cdef:
        public SIZE_t index
        public dict extra

        SIZE_t[:, :] triangles
        float[:, :] coordinates

        set vasculature_set_indices
        dict local_to_global_map


    def __cinit__(self, SIZE_t index,
                        float[:, :] local_coordinates,
                        SIZE_t[:, :] local_triangles,
                        set vasculature_set_indices,
                        dict local_to_global_map,
                        dict extra_data=None):

        self.index = index

        self.triangles = local_triangles
        self.coordinates = local_coordinates

        self.local_to_global_map = local_to_global_map
        self.vasculature_set_indices = vasculature_set_indices

        self.extra = extra_data


    def __reduce__(self):

        return (self.__class__, (self.index,
                                 np.asarray(self.coordinates),
                                 np.asarray(self.triangles),
                                 self.vasculature_set_indices,
                                 self.local_to_global_map,
                                 self.extra))


    @property
    def local_to_global_map(self):
        return self.local_to_global_map

    @property
    def coordinates_array(self):
        return np.asarray(self.coordinates)

    @property
    def vasculature_vertices(self):
        return np.asarray(sorted(self.vasculature_set_indices), dtype=np.uintp)

    @property
    def number_of_vertices(self):
        return len(self.coordinates)

    @property
    def number_of_triangles(self):
        return len(self.triangles)


    @property
    def triangle_array(self):
        return np.asarray(self.triangles)

    @property
    def edges(self):
        return np.asarray([(t[i], t[(i + 1) % 2]) for t in self.triangles for i in xrange(3)])

    cpdef float area_of_triangles(self, SIZE_t[:, :] triangles):

        cdef:
            float area = 0.0
            float[:] xyz1, xyz2, xyz3
            SIZE_t n

        for n in range(len(triangles)):

            xyz1 = self.coordinates[triangles[n, 0]]
            xyz2 = self.coordinates[triangles[n, 1]]
            xyz3 = self.coordinates[triangles[n, 2]]

            area += triangle_area(xyz1[0], xyz1[1], xyz1[2],
                                  xyz2[0], xyz2[1], xyz2[2],
                                  xyz3[0], xyz3[1], xyz3[2])
        return area

    cpdef float area_of_triangles_by_index(self, list indices):

        cdef:
            SIZE_t n
            SIZE_t[:, :] triangles = np.empty((len(indices), 3), dtype=np.uintp)

        for n in range(len(indices)):
            triangles[n] = self.triangles[indices[n]]

        return self.area_of_triangles(triangles)

    @property
    def area(self):
        return self.area_of_triangles(self.triangles)

    @property
    def vertex_neighbors(self):
        return _calculate_neighbors(self.triangles)

    @property
    def vertex_to_triangles(self):
        return _vertex_to_triangles(self.triangles)

    @property
    def edge_to_triangles(self):
        return _edge_to_triangles(self.triangles)

    cpdef void shrink(self, set indices_to_remove) except *:

        cdef:
            SIZE_t[:, :] new_triangles
            SIZE_t n, v

            set set_indices
            list sorted_indices
            dict old_to_new_map

        L.debug('Area Before Shrink: {}'.format(self.area))

        # subset of indices, subset of triangles
        new_triangles, set_indices = \
        subset_triangles_that_do_not_include_vertices(self.triangles, indices_to_remove)

        sorted_indices = sorted(set_indices)

        cdef float[:, :] new_coordinates = np.empty((len(sorted_indices), 3), dtype=np.float32)

        # slicing the coordinates we jump to a new reference system
        #self.coordinates = self.coordinates[sorted_indices]
        for n, v in enumerate(sorted_indices):
            new_coordinates[n] = self.coordinates[v]

        self.coordinates = new_coordinates

        # and the extra datasets

        if self.extra is not None:
            for dataset in self.extra['vertex'].values():
                dataset = [dataset[v] for v in sorted_indices]

        # therefore the triangle indices have to be remapped acccordingly
        old_to_new_map = {old: new for new, old in enumerate(sorted_indices)}

        for n in range(len(new_triangles)):

            new_triangles[n, 0] = old_to_new_map[new_triangles[n, 0]]
            new_triangles[n, 1] = old_to_new_map[new_triangles[n, 1]]
            new_triangles[n, 2] = old_to_new_map[new_triangles[n, 2]]

        self.triangles = new_triangles

        # the tricky part : channel the global vertices to this new map transition
        # first remove the removed indices using the existing local to global map
        remaining_vasculature_indices = map(self.local_to_global_map.__getitem__, set_indices)
        self.vasculature_set_indices = self.vasculature_set_indices.intersection(remaining_vasculature_indices)

        to_remove_local_keys = [k for k in self.local_to_global_map.keys() if k not in set_indices]

        for key in to_remove_local_keys:
            del self.local_to_global_map[key]

        # now change the map to reflect the new reference system
        new_local_to_global = \
        {old_to_new_map[old]: global_index for old, global_index in self.local_to_global_map.items()}

        self.local_to_global_map = new_local_to_global

        L.debug('Area After Shrink: {}'.format(self.area))
