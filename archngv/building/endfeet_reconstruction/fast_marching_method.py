"""fast marching method growing"""
import logging
from collections import deque

import numpy as np
from ngv_ctools.fast_marching_method import grow_waves_on_triangulated_surface
from scipy.spatial import cKDTree

from archngv.building.endfeet_reconstruction.groups import GroupedElements, group_elements
from archngv.exceptions import NGVError

L = logging.getLogger(__name__)


def _get_neighbors(vertex_neighbors, node):
    return vertex_neighbors.get_group_ids(node)


def _expanding_ring_neighborhoods(start_node, vertex_neighbors, max_level):
    """
    Generates expanding neighborhoods until it reaches max_level
    1-ring : neighbors of node
    2-ring : neighbors of neighbors of node without the previous neighbors
    .
    .
    """
    ring = {start_node}
    visited = {start_node}

    for _ in range(max_level):

        ring = set(
            neighbor
            for vertex in ring
            for neighbor in _get_neighbors(vertex_neighbors, vertex)
            if neighbor not in visited
        )
        visited |= ring

        yield ring


def _find_non_overlapping_mesh_nodes(
    closest_mesh_nodes, overlapping_groups, vertex_neighbors, distances
):
    """
    Places the overlapping endfeet ids to the same node at neighboring vertices where
    the 1-ring neighborhood of the new locations does not overlapin with the 1-ring
    locations of the already placed endfeet.

    Args:
        closest_mesh_nodes: array
            The closest mesh node for each endfoot position

        overlapping_groups: GroupedElements
            The data structure with the endfeet ids that overlap at a mesh node (group)

        vertex_neighbors: GroupedElements
            One group per vertex where its ids are the neighbors

        distances: array
            The distances of the endfeet ids to the closest mesh node

    Returns:
        closest_mesh_nodes: array
            The non overlapping closest mesh nodes to the endfeet array
    """
    occupied = set(closest_mesh_nodes)

    for mesh_node, endfeet_ids in overlapping_groups.items():

        if endfeet_ids.size <= 1:
            continue

        # expanding ring neighborhoods 1-ring, 2-ring etc.
        ring_neighborhood_generator = _expanding_ring_neighborhoods(mesh_node, vertex_neighbors, 5)

        ring = next(ring_neighborhood_generator)
        occupied |= ring

        endfeet_dst = distances[endfeet_ids]

        # leave the closest endfoot to be assigned to mesh_node
        # we will change the rest
        endfeet_list = deque(endfeet_ids[np.argsort(endfeet_dst)[::-1]][:-1])

        for ring in ring_neighborhood_generator:

            no_endfeet_left = False

            for vertex in ring:

                v_neighbors = _get_neighbors(vertex_neighbors, vertex)

                # we want to find a node where its 1-ring neighborhood
                # is not overlapping with the 1-ring neighborhood of the
                # other occupied points
                if not any(neighbor in occupied for neighbor in v_neighbors):

                    endfoot_id = endfeet_list.pop()
                    closest_mesh_nodes[endfoot_id] = vertex

                    # when found we add its 1-ring neighborhood and itself
                    # to the occupied vertices
                    occupied.add(vertex)
                    occupied |= set(v_neighbors)

                    if not endfeet_list:
                        no_endfeet_left = True
                        break

            if no_endfeet_left:
                break
        else:
            raise NGVError("Could not find neighboring available vertices to assign seeds")

        return closest_mesh_nodes


def _find_closest_mesh_nodes(endfeet_points, mesh_points, neighbors, nn_offsets):
    """for all endfeed points, find the closest points on the mesh"""
    L.info("Number of mesh points: %d", len(mesh_points))

    tree = cKDTree(mesh_points, leafsize=16, copy_data=False)

    # find the closest mesh indices to the endfeet targets
    distances, closest_mesh_nodes = tree.query(endfeet_points)

    # group overlapping nodes
    overlapping_groups = group_elements(closest_mesh_nodes)

    # No overlap
    if overlapping_groups.ids.size == overlapping_groups.groups.size:
        return closest_mesh_nodes

    L.info(
        "Multiple endfeet points converged to the same mesh node (%d, %d). Fixing...",
        overlapping_groups.ids.size,
        overlapping_groups.groups.size,
    )

    vertex_neighbors = GroupedElements(
        neighbors, nn_offsets, np.arange(len(nn_offsets) - 1, dtype=np.int64)
    )

    return _find_non_overlapping_mesh_nodes(
        closest_mesh_nodes, overlapping_groups, vertex_neighbors, distances
    )


def _mesh_to_flat_arrays(mesh):
    """Convert the mesh to three flat arrays which containt neighbors for each
    vertex, the offsets to access theese neighbors and the coordiantes of the vertices

    Args:
        mesh: openmesh Trimesh

    Returns:

        neighbors:
            Array of neighbors. The neighbors for the i-th vertex in the mesh are
            neighbors[nn_offsets[i]: nn_offsets[i + 1]]

        nn_offsets:
            See above

        vertex_coordiantes:
            The xyz coordinates of the vertices
    """
    neighbors = mesh.vv_indices()
    mask = neighbors >= 0

    nn_offsets = np.count_nonzero(mask.reshape(neighbors.shape), axis=1)

    nn_offsets = np.hstack(((0,), np.cumsum(nn_offsets))).astype(np.int64)
    neighbors = neighbors[mask].astype(np.int64)

    return neighbors, nn_offsets, mesh.points().astype(np.float32)


def fast_marching_eikonal_solver(mesh, seed_coordinates, cutoff_distance):
    """Fast Marching Eikonal Solver for unstructured grids.

    Propagates wavefronts from each source vertex. The wavefronts color
    the vertices they encounter and stop if another wavefront (group) has
    already beed colored by a neighboring vertex.

    Args:
        mesh: openmesh Trimesh

        seed_coordinates:
            The xyz coordinates of the initial seeds from which waves will be propagated
            on the surface of the mesh to color the vertices as the endfeet grow

        cutoff_distance:
            The maximum distance a wavefront can spread from its initial seed point

    Returns:
        v_group_indices:
            The group each vertex belongs to. The groups are corresponding to the positional
            indices of seed_coordinates except for -1 which represents the unvisited group.
            Unvisited vertices are the vertices that have not been reached by any wave.

        v_travel_times:
            The travel times per vertex that its respective seed wave needed to reach it.

        v_status:
            The status of the vertices. Can be 1: visited, 0: trial, -1: not visited
    """
    neighbors, nn_offsets, vertex_coordinates = _mesh_to_flat_arrays(mesh)

    seed_vertices = _find_closest_mesh_nodes(
        seed_coordinates, vertex_coordinates, neighbors, nn_offsets
    )

    v_group_indices, v_travel_times, v_status = grow_waves_on_triangulated_surface(
        neighbors, nn_offsets, vertex_coordinates, seed_vertices, cutoff_distance**2
    )

    return v_group_indices, v_travel_times, v_status
