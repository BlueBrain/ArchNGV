
import logging
import heapq
from itertools import groupby
from collections import namedtuple, deque

import numpy as np
cimport numpy as np
import scipy.sparse

from morphmath import vectorized_triangle_area


from .priority_heap cimport MaxPriorityHeap
from .priority_heap cimport PriorityHeapRecord
from .mesh_operations cimport find_surface_contours

L = logging.getLogger(__name__)

ctypedef np.npy_intp SIZE_t

def _remove_unconnected_triangles(endfoot):

    n_points = endfoot.number_of_vertices

    edges = endfoot.edges
    data = np.ones(len(edges), dtype=np.bool)

    M = scipy.sparse.csr_matrix((data, edges.T), shape=(n_points, n_points), dtype=np.bool)

    n_components, labels = scipy.sparse.csgraph.connected_components(M, directed=False, return_labels=True)

    if n_components > 1:

        to_remove = set()
        I = np.argsort(labels)

        it = groupby(I, key=lambda k: labels[k])

        _, current_it = it.next()

        current_vertices = list(current_it)

        for _, vit  in it:

            new_vertices = list(vit)

            if len(new_vertices) > len(current_vertices):

                to_remove.update(current_vertices)

                current_vertices = new_vertices


            else:

                to_remove.update(new_vertices)

        endfoot.shrink(to_remove)

    return endfoot


cpdef reduce_surface_area(object endfoot, float current_area, float target_area):
    """ Reduces current area to match target_area
    """
    cdef:
        MaxPriorityHeap heap
        PriorityHeapRecord r

        float current_time, oldest_time
        float[:] travel_times

        dict edge_triangles, neighbors, vertex_triangles

        set visited, to_remove, set_triangles_idx

        SIZE_t index, neighbor, oldest_neighbor, current_index

        tuple current_neighbors

        list ttimes

        frozenset current_edge


    L.debug('BEFORE Current Area: {}, Target Area: {}'.format(current_area, target_area))

    travel_times = endfoot.extra['vertex']['travel_times']

    heap = MaxPriorityHeap(len(travel_times))

    # map of each edge (undirected) to its incident triangles
    edge_triangles = endfoot.edge_to_triangles

    # find the contour of the mesh to start from the boundary
    visited = find_surface_contours(edge_triangles)

    # if there is no contour, return
    if not visited:
        L.warning('No contour found')
        return endfoot

    # store travel_time, index pairs according to their priority
    # with resepct to time. smallest to biggest travel time
    # we need to start removing the oldest vertices from the
    # endfoot point first
    #priority = [(travel_times[index], index) for index in visited]
    #heapq.heapify(priority)
    for index in visited:
        heap.push(index, travel_times[index])

    # map of each vertex to its neighboring vertices
    neighbors = endfoot.vertex_neighbors

    # the vertices that will be removed from the mesh in order to reduce
    # its total area
    to_remove = set()

    # follow the travel time backwards until the area is close to the target
    while current_area > target_area and not heap.is_empty():

        # oldest vertex in the contour
        heap.pop(&r)
        current_index = r.node_id
        current_time = r.value

        #current_time, current_index = priority.pop()
        to_remove.add(current_index)

        current_neighbors = tuple(neighbors[current_index].difference(visited))

        oldest_neighbor= -1
        oldest_time = 0.

        for neighbor in neighbors[current_index]:
            if neighbor not in visited:

                current_time = travel_times[neighbor]
            
                if current_time > oldest_time:

                    oldest_neighbor = neighbor
                    oldest_time = current_time


        if oldest_neighbor == -1:
            # No neighbors found, skip the rest and go to the next iteration
            continue

        visited.add(oldest_neighbor)

        # we are collapsing an edge
        current_edge = frozenset((current_index, oldest_neighbor))

        # the area subtracted is that of the incident triangles
        # to the edge that is removed. The triangles collapse, thus no
        # more contributing to the area of the mesh.
        set_triangles_idx = edge_triangles[current_edge]

        # if there are incident triangles to the edge
        if len(set_triangles_idx) > 0:

            # reduce the area accordingly
            current_area -= endfoot.area_of_triangles_by_index(list(set_triangles_idx))

            heap.push(oldest_neighbor, oldest_time)
            # add the new reduced contour vertex to the heap

        else:
            L.debug('Edge {} doesnt have incident triangles.'.format(current_edge))

    L.debug('N Vertices to be removed: {}'.format(len(to_remove)))


    # shrink the endfoot area
    endfoot.shrink(to_remove)

    L.debug('AFTER: Expected Area: {}, Actual Area: {}, Target Area: {}'.format(current_area, endfoot.area, target_area))

    return endfoot




def fit_area(endfoot, target_area):

    #L.debug('N_triangles: {}, N_indices: {}'.format(len(endfoot.triangles), len(endfoot.vasculature_vertices)))

    endfoot_area = endfoot.area

    try:
        assert 0. < target_area < endfoot_area

        L.info('Started surface area reduction: Target: {}, Current: {}'.format(target_area, endfoot_area))

        endfoot = reduce_surface_area(endfoot, endfoot_area, target_area)

        # remove all small triangles that fly around
        #endfoot = _remove_unconnected_triangles(endfoot)

    except AssertionError:

         L.info('Aborted surface area reduction')

    L.info('Endfoot Area Fitting completed. Target: {}, Current: {}'.format(target_area, endfoot.area))
    return endfoot


