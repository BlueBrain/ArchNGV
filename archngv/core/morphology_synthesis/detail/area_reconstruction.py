from collections import deque
import numpy as np
import trimesh
import logging
import igraph


def create_face_graph_from_mesh(mesh):
    """ Creates a graph of the mesh's triangular faces
    """
    face_graph = igraph.Graph()
    face_graph.add_vertices(list(range(len(mesh.faces))))
    face_graph.add_edges(mesh.face_adjacency)

    return face_graph


def expand_sites(face_graph, starting_face_idx, area_faces, area_threshold, return_areas=False):

    n_faces = len(face_graph.vs)
    n_clusters = len(starting_face_idx)

    cluster_faces = [deque([start_index]) for start_index in starting_face_idx]
    cluster_neighbors = [[start_index] for start_index in starting_face_idx]

    cluster_areas = np.array([area_faces[start_index] for start_index in starting_face_idx])

    visited_faces = set(starting_face_idx.tolist())

    k = 0
    non_empty = n_clusters

    while non_empty > 0:

        q = cluster_neighbors[k]

        if q:

            new_q = deque()

            while q:

                face_index = q.pop()

                face_neighbors = face_graph.neighbors(face_index)
                face_neighbors = filter(lambda nn: nn not in visited_faces, face_neighbors)

                for face_id in face_neighbors:

                    new_cluster_area = cluster_areas[k] + area_faces[face_id]

                    if new_cluster_area < area_threshold:

                        visited_faces.add(face_id)

                        cluster_faces[k].append(face_id)
                        cluster_areas[k] = new_cluster_area

                        new_q.append(face_id)

                    else:
                        break

            if new_q:

                cluster_neighbors[k] = new_q

            else:

                non_empty -= 1

        k = (k + 1) % n_clusters

    if return_areas:

        return cluster_faces, cluster_areas

    else:

        return cluster_faces


def create_endfeet_areas(vasculature_mesh, surface_targets, options):

    area_threshold = options['max_endfoot_area']

    face_graph = create_face_graph_from_mesh(vasculature_mesh)

    area_faces = vasculature_mesh.area_faces

    _, _, seed_face_idx = vasculature_mesh.nearest.on_surface(surface_targets)

    face_idx_per_endfoot, endfeet_areas = \
    expand_sites(face_graph, seed_face_idx, area_faces, area_threshold, return_areas=True)

    return face_idx_per_endfoot, endfeet_areas
