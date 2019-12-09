'''fast marching method growing'''
import collections
import logging
import numpy as np

from scipy.spatial import cKDTree
from archngv_building.endfeet_reconstruction import fmm_growing

L = logging.getLogger(__name__)

FMM_FAR = -1
FMM_KNOWN = 1
UNKNOWN_GROUP = -1


def _find_closest_mesh_nodes(endfeet_points, mesh_points):
    '''for all endfeed points, find the closest points on the mesh'''
    L.info('Number of mesh points: %s', len(mesh_points))

    tree = cKDTree(mesh_points, leafsize=16, copy_data=False)

    # find the closest mesh indices to the endfeet targets
    _, mesh_nodes = tree.query(endfeet_points)
    set_occupied_nodes = set(mesh_nodes)

    if len(set_occupied_nodes) != mesh_nodes.size:
        L.info('Multiple endfeet points converged to the same mesh node (%s, %s). Fixing...',
            len(set_occupied_nodes), mesh_nodes.size)

        endfeet_per_node = collections.defaultdict(set)
        for endfoot_index, mesh_node in enumerate(mesh_nodes):
            endfeet_per_node[mesh_node].add(endfoot_index)

        iter_multiples_only = filter(lambda e: len(e) > 1, endfeet_per_node.values())

        for endfeet_set in iter_multiples_only:
            assert len(endfeet_set) < 5, 'Sparse mesh. Too many endfeet close to the same mesh node.'
            endfeet_list = list(endfeet_set)
            for endfoot_index in endfeet_list[1:]:
                _, nearest_neighbors = tree.query(endfeet_points[endfoot_index], k=10)
                for nearest_neighbor in nearest_neighbors:
                    if nearest_neighbor not in set_occupied_nodes:
                        mesh_nodes[endfoot_index] = nearest_neighbor
                        set_occupied_nodes.add(nearest_neighbor)
                        break
                else:
                    raise Exception('Fixing closeby points failed.')

    return mesh_nodes


def _groups(v_group_index):
    '''transform v_group_index (storing vertex -> group information) into inverse

                    0 1 2 3 4 5 6  7  8  # (implicit vertex index)
     v_group_index  3 3 3 2 2 1 0 -1 -1  # group; -1 means not specified
     ->
               0  1  2  3  4  5  6  # index
        idx = [6, 5, 3, 4, 0, 1, 2, ]
                   0  1  2  3      # implicit group index
        offsets = [0, 1, 2, 4, 7]  # offsets into above
    '''
    # group vertices with same seed, note: casting to uint to have -1 sort at the end
    values = v_group_index.astype(np.uint)
    idx = np.argsort(values)[:np.count_nonzero(v_group_index >= 0)]
    _, offsets = np.unique(values, return_counts=True)
    offsets = np.cumsum(offsets)
    offsets[1:] = offsets[:-1]
    offsets[0] = 0
    return idx, offsets


class FastMarchingEikonalSolver:
    """ Fast Marching Eikonal Solver for unstructured grids.

    Vertices start as FMM_FAR and as they are found in the neighborhood of the
    wavefront they become FMM_TRIAL. Finally, as the travel time of the
    wavefront is updated they become KNOWN and by extension frozen for the rest
    of the calculation.
    """
    def __init__(self, mesh, target_points, cutoff_distance):
        self.n_seeds = len(target_points)
        self.n_vertices = mesh.n_vertices()

        self.squared_cutoff_distance = cutoff_distance * cutoff_distance

        L.info('Copying neighbor information...')
        L.info('assign the neighbors for each vertex')
        self.neighbors, self.v_xyz, self.nn_offsets = self._assign_vertex_neighbors(mesh)

        # the label vertex that acts as starting point of the growth
        L.info('find_closest_mesh_nodes')
        self.group_labels = _find_closest_mesh_nodes(target_points,
                                                     mesh.points().astype(np.float32))
        self.v_status, self.v_group_index, self.v_travel_time = self._assign_point_sources()

    def _assign_point_sources(self):
        '''generate vertex groups and their initial travel times'''
        L.info('assign point sources')
        v_status = np.full(self.n_vertices, FMM_FAR, dtype=np.long)
        v_group_index = np.full(self.n_vertices, UNKNOWN_GROUP, dtype=np.long)
        v_travel_time = np.full(self.n_vertices, np.inf, dtype=np.float32)

        group_index = np.arange(self.n_seeds)
        v_index = self.group_labels[group_index]
        v_group_index[v_index] = group_index
        v_status[v_index] = FMM_KNOWN
        v_travel_time[v_index] = 0.0

        return v_status, v_group_index, v_travel_time

    @staticmethod
    def _assign_vertex_neighbors(mesh):
        '''assign the neighbors for each vertex'''
        neighbors = mesh.vv_indices()
        mask = neighbors >= 0
        nn_offsets = np.count_nonzero(mask.reshape(neighbors.shape), axis=1)
        nn_offsets = np.hstack(((0, ), np.cumsum(nn_offsets))).astype(np.long)
        neighbors = neighbors[mask].astype(np.long)
        v_xyz = mesh.points().astype(np.float32)

        return neighbors, v_xyz, nn_offsets

    def marks(self):
        ''' Returns the group id for each vertex in the vertices array.

        A group represents a wavefront that started propagated from each seed source vertex
        '''
        return np.asarray(self.v_group_index)

    def travel_times(self):
        '''Returns the travel time that the closest wavefront needed to reach each vertex'''
        return np.asarray(self.v_travel_time)

    def solve(self):
        ''' Runs the eikonal solver

        Propagates wavefronts from each
        source vertex. The wavefronts color the vertices they encounter and
        stop if another wavefront (group) has already beed colored a
        neighboring vertex. '''
        return fmm_growing.solve(self.squared_cutoff_distance,
                                 self.nn_offsets,
                                 self.v_travel_time,
                                 self.neighbors,
                                 self.v_xyz,
                                 self.v_status,
                                 self.group_labels,
                                 self.v_group_index,
                                 )

    def groups(self):
        ''' transform v_group_index (storing vertex -> group information) into inverse
        Returns:
            vertices: array[int, (n_vertices,)]
                    All the vertices with a per group ordering.
                group_offsets: array[int, (n_seeds + 1)]
                    An offset array which allows us to extract the vertices for each group.
                    For example the vertices that belong to group 5 are
                        vertices[group_offsets[5]: group_offsets[6]]
        '''
        return _groups(self.v_group_index)
