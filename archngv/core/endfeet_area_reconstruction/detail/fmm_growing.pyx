# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False

import time
import logging
import numpy as np
cimport numpy as np

from .priority_heap cimport MinPriorityHeap
from .priority_heap cimport PriorityHeapRecord

from libc.math cimport INFINITY, isinf, ceil, sqrt
from .local_solvers cimport local_solver_2D 

from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map

from libcpp.algorithm cimport sort
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdlib cimport malloc, free
L = logging.getLogger(__name__)

from scipy.spatial import cKDTree


from cpython.array cimport array, clone


ctypedef np.npy_long SIZE_t




cdef class FastMarchingEikonalSolver:

    cdef:
        # distance matrix
        float[:, :] Q
        float squared_cutoff_distance

        SIZE_t FAR, TRIAL, KNOWN
        SIZE_t NONE, n_seeds, n_vertices

        MinPriorityHeap trial_heap

        SIZE_t[:] group_labels

        # Vertex neighbors stored in linear arrays
        vector[SIZE_t] neighbors

        # neighbors for vertex i: neighbors[n_offsets[i]: n_offsets[i + 1]]
        SIZE_t[:] nn_offsets

        # Vertex attributes
        #################################
        float[:, :]     v_xyz           # coordinates
        SIZE_t[:]       v_status        # FAR, TRIAL, KNOWN
        float[:]        v_travel_time   # Time of wave propagation
        SIZE_t[:]       v_group_index   # index to group_labels

    def _initialization(self, SIZE_t n_vertices):

        self.FAR = -1
        self.TRIAL = 0
        self.KNOWN = 1
        self.NONE = -1

        # all vertices start as FAR
        self.v_status = np.full(n_vertices, self.FAR, dtype=np.long)

        # no groups in the beginning
        self.v_group_index = np.full(n_vertices, self.NONE, dtype=np.long)

        # and travel time to infinity
        self.v_travel_time = np.full(n_vertices, np.inf, dtype=np.float32)

        # coordinates empty
        self.v_xyz = np.empty((n_vertices, 3), dtype=np.float32)

        # priority heap that sorts vertices according to their stored
        # travel times
        self.trial_heap = MinPriorityHeap(self.n_vertices)

        # distance matrix
        # TODO: this should be external
        self.Q = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)

        # neighbor offsets
        self.nn_offsets = np.zeros(n_vertices + 1, dtype=np.long)

    def __cinit__(self, object mesh,
                        float[:, :] target_points,
                        float cutoff_distance):

        """ Fast Marching Eikonal Solver for unstructured grids. Vertices start
        as FAR and as they are found in the neighborhood of the wavefront they
        become TRIAL. Finally, as the travel time of the wavefront is updated
        they become KNOWN and by extension frozen for the rest of the calculation.
        """
        cdef:
            SIZE_t v_index, group_index, offset

        self.n_seeds = len(target_points)
        self.n_vertices = mesh.n_vertices()

        self.squared_cutoff_distance = cutoff_distance * cutoff_distance
        L.info('Cutoff distance: {}'.format(cutoff_distance))

        self._initialization(self.n_vertices)

        # the label vertex that acts as starting point of the growth
        self.group_labels = find_closest_mesh_nodes(target_points, mesh.points().astype(np.float32))

        #####################################################
        # assign the neighbors for each vertex
        #####################################################

        L.info('Copying neighbor information...')

        offset = 0
        self.neighbors.reserve(self.n_vertices) # start with reasonable reservation
        for v_handle in mesh.vertices():

            v_index = v_handle.idx()
            for n_handle in mesh.vv(v_handle):
                self.neighbors.push_back( n_handle.idx() )
                offset += 1
            self.nn_offsets[v_index + 1] = offset

            # node coordinates
            self.v_xyz[v_index, 0],\
            self.v_xyz[v_index, 1],\
            self.v_xyz[v_index, 2] = mesh.point(v_handle)

        #####################################################
        # assign point sources
        #####################################################
        for group_index in range(self.n_seeds):

            # the group starting vertex index
            v_index = self.group_labels[group_index]

            # unique enumerating id to seed vertex
            self.v_group_index[v_index] = group_index

            # vertices are frozen as seed points
            self.v_status[v_index] = self.KNOWN

            # and trave time set to zero
            self.v_travel_time[v_index] = 0.0

    cdef inline bint close_to_seed(self, SIZE_t ind1, SIZE_t ind2) nogil:

        cdef float d2

        d2 = (self.v_xyz[ind1, 0] - self.v_xyz[ind2, 0]) ** 2

        if d2 > self.squared_cutoff_distance:
            return False

        d2 += (self.v_xyz[ind1, 1] - self.v_xyz[ind2, 1]) ** 2

        if d2 > self.squared_cutoff_distance:
            return False

        d2 += (self.v_xyz[ind1, 2] - self.v_xyz[ind2, 2]) ** 2

        return d2 < self.squared_cutoff_distance

    cdef inline float local_update(self, SIZE_t ind) nogil:
        """ Update the vertex value by solving the eikonal
        equation using the first order discretization of the gradient

        """
        cdef:
            float TA, TB, TC, min_value = self.v_travel_time[ind]
            SIZE_t n, nb1, nb2
            SIZE_t nn_start = self.nn_offsets[ind]
            SIZE_t n_neighbors = self.nn_offsets[ind + 1] - nn_start

        # find a triangle with nodes of known values
        # and update the traveling time at vertex C
        for n in range(n_neighbors):
            if n == n_neighbors - 1:

                nb1 = self.neighbors[nn_start + n]
                # last edge cycled to the start
                nb2 = self.neighbors[nn_start]

            else:
                # consecutive pair
                nb1 = self.neighbors[nn_start + n]
                nb2 = self.neighbors[nn_start + n + 1]

            TA = self.v_travel_time[nb1]
            TB = self.v_travel_time[nb2]

            if not (isinf(TA) and isinf(TB)):

                # ensure TB > TA with the ol good switcheroo
                if TB >= TA:
                    TC = local_solver_2D(self.v_xyz[nb1, 0], self.v_xyz[nb1, 1], self.v_xyz[nb1, 2],
                                         self.v_xyz[nb2, 0], self.v_xyz[nb2, 1], self.v_xyz[nb2, 2],
                                         self.v_xyz[ind, 0], self.v_xyz[ind, 1], self.v_xyz[ind, 2],
                                         TA, TB,
                                         self.Q[0, 0], self.Q[1, 1], self.Q[2, 2])
                else:

                    TC = local_solver_2D(self.v_xyz[nb2, 0], self.v_xyz[nb2, 1], self.v_xyz[nb2, 2],
                                         self.v_xyz[nb1, 0], self.v_xyz[nb2, 1], self.v_xyz[nb1, 2],
                                         self.v_xyz[ind, 0], self.v_xyz[ind, 1], self.v_xyz[ind, 2],
                                         TB, TA,
                                         self.Q[0, 0], self.Q[1, 1], self.Q[2, 2])

                if TC < min_value:
                    min_value = TC

        return min_value

    cdef inline void update_neighbors(self, SIZE_t vertex_index) nogil except *:
        """ Update the values of the one ring neighbors of a vertex
        """
        cdef:
            SIZE_t neighbor_status
            float new_travel_time
            SIZE_t n, nv
            SIZE_t nn_start = self.nn_offsets[vertex_index]
            SIZE_t n_neighbors = self.nn_offsets[vertex_index + 1] - nn_start

        for n in range(n_neighbors):

            nv = self.neighbors[nn_start + n]
            neighbor_status = self.v_status[nv]

            # if the neighbor value has not been finalized (FAR, TRIAL)
            if  neighbor_status != self.KNOWN:

                # find the travel time of the wave to the
                # neighbor vertex in the ring
                new_travel_time = self.local_update(nv)
                self.v_travel_time[nv] = new_travel_time

                # otherwise add in the priority queue with the travel time
                # as priority. It starts as a trial vertex.
                if neighbor_status == self.FAR:
                    
                    if self.close_to_seed(nv, self.group_labels[self.v_group_index[vertex_index]]):

                        self.v_group_index[nv] = self.v_group_index[vertex_index]
                        self.trial_heap.push(nv, new_travel_time)
                        self.v_status[nv] = self.TRIAL


    cpdef void solve(self) except *:
        """ Solves the eikonal equations using the fast marching method
        """

        cdef:
            PriorityHeapRecord record
            float travel_time
            SIZE_t vertex_index

        L.info('Updating source neighbors...')

        with nogil:

            # fist iterate over the known nodes and calculate the
            # travel time to the neighbors
            for vertex_index in range(self.n_vertices):
                if self.v_group_index[vertex_index] != self.NONE:
                    self.update_neighbors(vertex_index)

        assert not self.trial_heap.is_empty()

        L.info('Started Growing...')

        with nogil:
            # expand in a breadth first manner from the smallest
            # distance node and update the travel times for the 
            # propagation of the wavefront
            while not self.trial_heap.is_empty():

                # min travel time vertex
                self.trial_heap.pop(&record)
                travel_time = record.value
                vertex_index = record.node_id
        
                if self.v_status[vertex_index] != self.KNOWN:

                    self.v_status[vertex_index] = self.KNOWN
                    self.v_travel_time[vertex_index] = travel_time

                self.update_neighbors(vertex_index)

        L.info('Solve Completed')

    cpdef marks(self):
        return np.asarray(self.v_group_index)

    cpdef groups(self):

        cdef:
            SIZE_t value, offset, vertex_index, group_index 
            SIZE_t[:] idx = np.empty(self.n_vertices, dtype=np.intp)
            SIZE_t[:] offsets = np.zeros(self.n_seeds + 1, dtype=np.intp)

            list seed_registry = [set() for _ in xrange(self.n_seeds)]

            set elements

        # group vertices with same seed
        for vertex_index in range(self.n_vertices):
            group_index = self.v_group_index[vertex_index]
            if group_index != self.NONE:
                seed_registry[group_index].add(vertex_index)

        offset = 0

        # get the groups and store them in a linear
        # array using offsets for group sizes
        for label_index in range(self.n_seeds):
            for value in seed_registry[label_index]:
                idx[offset] = value
                offset += 1
            offsets[label_index + 1] = offset

        assert offsets[label_index + 1] == offset, "Offset mismatch: {}, {}".format(offsets[label_index + 1], offset)

        return (np.asarray(self.group_labels),
                np.asarray(idx),
                np.asarray(offsets))

    cpdef travel_times(self):
        return np.asarray(self.v_travel_time)

################################################

cpdef SIZE_t[:] find_closest_mesh_nodes(float[:, :] endfeet_points, float[:, :] mesh_points):

    cdef:

        SIZE_t n, mesh_index, endfoot_index
        SIZE_t[:] seed_idx
        SIZE_t[:] unique_indices

        dict registry
        list duplicates
        set set_unique_indices, endfeet_indices

    L.info('Number of mesh points: {}'.format(len(mesh_points)))

    tree = cKDTree(mesh_points, leafsize=16, copy_data=False)

    # find the closest mesh indices to the endfeet targets
    seed_idx = tree.query(endfeet_points)[1]
    unique_indices = np.unique(seed_idx)

    if unique_indices.size != seed_idx.size:

        L.info('Multiple endfeet points converged to the same mesh node ({}, {}). Fixing...'.format(unique_indices.size, seed_idx.size))

        registry = {mesh_index: set() for mesh_index in unique_indices}

        for endfoot_index, mesh_index in enumerate(seed_idx):
            registry[mesh_index].add(endfoot_index)

        duplicates = [item for item in registry.items() if len(item[1]) > 1]

        set_unique_indices = set([n for n in unique_indices])

        for mesh_index, endfeet_indices in duplicates:

            assert len(endfeet_indices) < 5, 'Sparse mesh. Finer subdivisions required.'

            for endfoot_index in list(endfeet_indices)[1:]:

                nearest_neighbors = tree.query(endfeet_points[endfoot_index], k=10)[1]

                found = False

                for nearest_neighbor in nearest_neighbors:
                    if nearest_neighbor not in set_unique_indices:

                        seed_idx[endfoot_index] = nearest_neighbor
                        set_unique_indices.add(nearest_neighbor)
                        found = True
                        break

                assert found, 'Fixing closeby points failed.'

    return seed_idx
