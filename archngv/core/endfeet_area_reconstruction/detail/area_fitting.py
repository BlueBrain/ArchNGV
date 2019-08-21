'''area fitting'''
from heapq import heappush, heappop
import logging


L = logging.getLogger(__name__)


def fit_area(endfoot, target_area):
    '''take and endfoot, and try to reduce it to target_area'''
    try:
        assert 0. < target_area < endfoot.area
        L.info('Started surface area reduction: Target: %s, Current: %s',
               target_area, endfoot.area)

        to_remove = _reduce_surface_area(endfoot,
                                         target_area,
                                         endfoot.extra['vertex']['travel_times'])

        L.debug('N Vertices to be removed: %d', len(to_remove))
        endfoot.shrink(to_remove)
        L.debug('AFTER: Actual Area: %s, Target Area: %s',
                endfoot.area, target_area)
    except AssertionError:
        L.info('Aborted surface area reduction')

    L.info('Endfoot Area Fitting completed. Target: %s, Current: %s',
           target_area, endfoot.area)
    return endfoot


def _find_border_edges(edge_triangles):
    '''find edges of triangles on the border'''
    return set(e
               for edge, triangles in edge_triangles.items()
               if len(triangles) == 1
               for e in edge
               )


def _reduce_surface_area(endfoot, target_area, travel_times):
    """ Reduces current area to match target_area """
    edge_triangles = endfoot.edge_to_triangles
    all_neighbors = endfoot.vertex_neighbors

    # find the contour of the mesh to start from the boundary
    visited = _find_border_edges(edge_triangles)

    # if there is no contour, return
    if not visited:
        return set()

    # store travel_time, index pairs according to their priority
    # with resepct to time. smallest to biggest travel time
    # we need to start removing the oldest vertices from the
    # endfoot point first
    heap = []
    for index in visited:
        heappush(heap, (-travel_times[index], index))

    # the vertices that will be removed from the mesh in order to reduce its total area
    to_remove = set()

    # follow the travel time backwards until the area is close to the target
    current_area = endfoot.area
    while current_area > target_area and heap:
        # oldest vertex in the contour
        _, index = heappop(heap)
        to_remove.add(index)

        neighbors = all_neighbors[index] - visited
        if not neighbors:
            continue

        # we are collapsing an edge

        oldest_time, oldest_neighbor = max((travel_times[neighbor], neighbor)
                                           for neighbor in neighbors)

        visited.add(oldest_neighbor)

        # the area subtracted is that of the incident triangles
        # to the edge that is removed. The triangles collapse, thus no
        # more contributing to the area of the mesh.
        incident_triangles = edge_triangles[frozenset((index, oldest_neighbor))]

        # if there are incident triangles to the edge
        if incident_triangles:
            # reduce the area accordingly
            current_area -= endfoot.area_of_triangles_by_index(list(incident_triangles))
            heappush(heap, (-oldest_time, oldest_neighbor))

    return to_remove
