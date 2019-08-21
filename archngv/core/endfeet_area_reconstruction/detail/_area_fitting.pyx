cimport numpy as np

from .priority_heap cimport MaxPriorityHeap
from .priority_heap cimport PriorityHeapRecord
from .mesh_operations cimport find_surface_contours

ctypedef np.npy_intp SIZE_t


cpdef set reduce_surface_area(object endfoot,
                              float target_area,
                              float[:] travel_times):
    """ Reduces current area to match target_area """
    cdef:
        MaxPriorityHeap heap
        PriorityHeapRecord r

        float current_time, oldest_time, current_area

        dict edge_triangles, neighbors

        set visited, to_remove, set_triangles_idx

        SIZE_t index, neighbor, oldest_neighbor

        list ttimes

        frozenset current_edge

    current_area = endfoot.area

    heap = MaxPriorityHeap(len(travel_times))

    # map of each edge (undirected) to its incident triangles
    edge_triangles = endfoot.edge_to_triangles

    # find the contour of the mesh to start from the boundary
    visited = find_surface_contours(edge_triangles)

    # if there is no contour, return
    if not visited:
        return set()

    # store travel_time, index pairs according to their priority
    # with resepct to time. smallest to biggest travel time
    # we need to start removing the oldest vertices from the
    # endfoot point first
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
        index = r.node_id
        current_time = r.value

        to_remove.add(index)

        oldest_neighbor= -1
        oldest_time = 0.

        for neighbor in neighbors[index]:
            if neighbor in visited:
                continue

            current_time = travel_times[neighbor]
            if current_time > oldest_time:
                oldest_neighbor = neighbor
                oldest_time = current_time

        if oldest_neighbor == -1:
            # No neighbors found, skip the rest and go to the next iteration
            continue

        visited.add(oldest_neighbor)

        # we are collapsing an edge
        current_edge = frozenset((index, oldest_neighbor))

        # the area subtracted is that of the incident triangles
        # to the edge that is removed. The triangles collapse, thus no
        # more contributing to the area of the mesh.
        set_triangles_idx = edge_triangles[current_edge]

        # if there are incident triangles to the edge
        if len(set_triangles_idx) > 0:
            # reduce the area accordingly
            current_area -= endfoot.area_of_triangles_by_index(list(set_triangles_idx))
            heap.push(oldest_neighbor, oldest_time)
    return to_remove
