'''endfoot analysis'''
from collections import defaultdict
import logging
import numpy as np
from archngv_building.endfeet_reconstruction import endfoot

L = logging.getLogger(__name__)


def create_endfoot_from_global_data(index,
                                    all_coordinates,
                                    all_triangles,
                                    endfoot_vertices):
    '''Create an endfoot object in its local coordinate system

    (containing only vertices and triangles that correspond to it) using the
    global datasets of all coordinates, triangles and vertices per endfoot
    '''
    kept_triangles = endfoot.subset_triangles_that_include_vertices(
        all_triangles, set(endfoot_vertices))

    coordinates, triangles, local_to_global_map = _arrange_indices(
        kept_triangles, all_coordinates, all_triangles)

    return Endfoot(index,
                   coordinates,
                   triangles,
                   set(endfoot_vertices),
                   local_to_global_map)


class Endfoot(object):
    '''Endfoot'''
    def __init__(self,
                 index,
                 coordinates,
                 triangles,
                 vasculature_set_indices,
                 local_to_global_map,
                 extra_data=None):
        self.index = index
        self.coordinates = coordinates
        self.triangles = triangles
        self.vasculature_set_indices = vasculature_set_indices
        self.local_to_global_map = local_to_global_map
        self.extra = extra_data

    def __reduce__(self):
        '''for pickling'''
        return (self.__class__, (self.index,
                                 np.asarray(self.coordinates),
                                 np.asarray(self.triangles),
                                 self.local_to_global_map,
                                 self.extra))

    @property
    def coordinates_array(self):
        '''vertices in the endfoot'''
        return np.asarray(self.coordinates)

    @property
    def vasculature_vertices(self):
        '''vertices in this endfoot in the global coordinate system of the vasculature'''
        return np.asarray(sorted(self.vasculature_set_indices), dtype=np.uintp)

    @property
    def number_of_vertices(self):
        '''Number of endfoot mesh vertices'''
        return len(self.coordinates)

    @property
    def number_of_triangles(self):
        '''Number of triangles in endfoot mesh'''
        return len(self.triangles)

    @property
    def triangle_array(self):
        '''Array of endfoot mesh triangles'''
        return np.asarray(self.triangles)

    @property
    def edges(self):
        '''Edges connecting the vertices of the endfoot mesh'''
        return np.asarray([(t[i], t[(i + 1) % 2])
                           for t in self.triangles
                           for i in range(3)])

    @property
    def area(self):
        '''Area of the endfoot'''
        return self._area_of_triangles(self.triangles)

    @property
    def vertex_neighbors(self):
        '''Returns dict with vertices as keys and their neighbors as values'''
        ns = defaultdict(set)
        for v1, v2, v3 in self.triangles:
            for key, neighs in ((v1, (v2, v3)),
                                (v2, (v3, v1)),
                                (v3, (v1, v2))):
                ns[key].update(neighs)
        return dict(ns)

    @property
    def vertex_to_triangles(self):
        '''Dict of vertices as keys and the triangles they are part of for each key'''
        v2t = defaultdict(set)
        for index, triangle in enumerate(self.triangles):
            for vertex in triangle:
                v2t[vertex].add(index)
        return dict(v2t)

    @property
    def edge_to_triangles(self):
        '''Dict of frozensets of edges as keys and triangles that each edge is part of as values

        Note: Frozensets are used so that directionality does not matter
        '''
        et = defaultdict(set)
        for index, (v1, v2, v3) in enumerate(self.triangles):
            for c1, c2 in ((v1, v2), (v2, v3), (v3, v1)):
                et[frozenset((c1, c2))].add(index)
        return dict(et)

    def _area_of_triangles(self, triangles):
        '''calculate the area of the triangles indexed by triangles'''
        ab = (self.coordinates[triangles[:, 0]] - self.coordinates[triangles[:, 1]])
        ac = (self.coordinates[triangles[:, 0]] - self.coordinates[triangles[:, 2]])
        return np.linalg.norm(np.cross(ab, ac), axis=1).sum() / 2.

    def area_of_triangles_by_index(self, indices):
        '''calculate the area of the triangles referenced by local `indices`'''
        return self._area_of_triangles(self.triangles[indices, :])

    def shrink(self, indices_to_remove):
        '''Shrink the endfoot mesh

        removing the indices_to_remove and adapt the rest of the datasets
        accordingly ensuring the maps to the global maps are consistent
        '''
        L.debug('Area Before Shrink: %s', self.area)

        kept_triangles = endfoot.subset_triangles_that_do_not_include_vertices(
            self.triangles, indices_to_remove)

        extras = None
        if self.extra is not None and 'vertex' in self.extra:
            extras = self.extra['vertex']

        self.coordinates, self.triangles, local_to_global_map = _arrange_indices(
            kept_triangles, self.coordinates, self.triangles, extras)

        self.local_to_global_map = {k: self.local_to_global_map[v]
                                    for k, v in local_to_global_map.items()}

        self.vasculature_set_indices = set(self.local_to_global_map[i]
                                           for i in np.unique(self.triangles))

        L.debug('Area After Shrink: %s', self.area)


def _arrange_indices(kept_triangles, coordinates, triangles, extras=None):
    '''keep subset of trianges and coordinates based on `kept_triangles`

    Args:
        kept_triangles(iterable): indices of triangles to keep
        coordinates(np.array): vertices of triangles
        triangles(np.array): triangle index into coordinates array
        extras(dict): values of this dictionary are indexed by triangled
        index, and are updated to only have `kept_triangles`
        Warning: *MODIFIED IN PLACE*
    '''
    kept_triangles = sorted(kept_triangles)
    triangles = triangles[kept_triangles, :]
    used_vertices = np.unique(triangles)

    coordinates = coordinates[used_vertices]

    # therefore the triangle indices have to be remapped acccordingly
    global_to_local_map = {old: new for new, old in enumerate(used_vertices)}

    for n in range(len(triangles)):
        triangles[n, 0] = global_to_local_map[triangles[n, 0]]
        triangles[n, 1] = global_to_local_map[triangles[n, 1]]
        triangles[n, 2] = global_to_local_map[triangles[n, 2]]

    # finally we need to keep a record of the mapping between the local and global
    # reference system in order to be able to go back later
    local_to_global_map = dict(enumerate(used_vertices))

    if extras:
        for k in extras:
            extras[k] = extras[k][used_vertices]

    return coordinates, triangles, local_to_global_map
