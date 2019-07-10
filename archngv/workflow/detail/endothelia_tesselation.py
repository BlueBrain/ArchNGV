import os
import stl
import sys
import h5py
import logging
import openmesh
from collections import OrderedDict
import numpy as np
import multiprocessing
from scipy import stats
from scipy.spatial import cKDTree

# TODO: these imports below are not pointing to anything
from archngv.synthesis.endfeet_reconstruction.endfoot import create_endfoot_from_global_data

from archngv.synthesis.endfeet_reconstruction.fmm_growing import FastMarchingEikonalSolver
from archngv.synthesis.endfeet_reconstruction.fmm_growing import find_closest_mesh_nodes_from_points
from archngv.synthesis.endfeet_reconstruction import io


L = logging.getLogger(__name__)


if __name__ == '__main__':

    mesh_path = sys.argv[1]
    input_path = sys.argv[2]

    threshold_radius = float(sys.argv[3])
    output_filepath = sys.argv[4]

    input_points = np.loadtxt(input_path)

    L.info('Loading mesh')

    mesh = openmesh.read_trimesh(mesh_path)
    mesh_points = mesh.points().astype(np.double)
    mesh_triangles = mesh.face_vertex_indices().astype(np.uintp)

    ####################################################################################

    L.info('fmm started.')

    #threshold_radius = 10.

    L.info('Estimating closest nodes to endfeet targets to use as seed vertices.')
    target_indices = find_closest_mesh_nodes_from_points(input_points.astype(np.float32), mesh_points.astype(np.float32))

    assert np.unique(target_indices).size == len(input_points), 'Closeby points that refer to the same vertex'

    solver = FastMarchingEikonalSolver(
                                           mesh, 
                                           target_indices,
                                           threshold_radius
                                      )
    solver.solve()

    travel_times = solver.travel_times()
    mark_indices, mark_offsets = solver.groups()

    L.info('fmm completed')

    ####################################################################################

    L.info('Generating vertex dict')

    v2tr = [set() for v in xrange(len(mesh_points))]

    for (v0, v1, v2) in mesh_triangles:
        s = frozenset((v0, v1, v2))
        v2tr[v0].add(s)
        v2tr[v1].add(s)
        v2tr[v2].add(s)

    L.info('Writing meshes')


    with h5py.File(output_filepath, 'w') as fd:

        metadata = fd.create_group('metadata')

        metadata.attrs['object_type'] = 'endothelial_cell_mesh'
        metadata.attrs['vasculature_mesh'] = os.path.basename(mesh_path)
        metadata.attrs['point_dataset'] = os.path.basename(input_path)
        metadata.attrs['number_of_points'] = len(input_points)
        metadata.attrs['method_of_point_generation'] = 'Poisson Disk Sampling'

        meshes = fd.create_group('objects')

        for input_index in xrange(len(input_points)):

            set_mesh_vertices = set(mark_indices[mark_offsets[input_index]: mark_offsets[input_index + 1]])
            triangles = set(triangle for v in set_mesh_vertices for triangle in v2tr[v])

            set_mesh_vertices = set(v for tr in triangles for v in tr)

            g2l = OrderedDict([(old, new) for new, old in enumerate(set_mesh_vertices)])

            triangles = [[g2l[v] for v in triangle] for triangle in triangles]

            global_ref_idx = g2l.keys()

            input_point = input_points[input_index]

            points = mesh_points[global_ref_idx]

            # h5 specifics

            mesh_group = meshes.create_group('mesh_{}'.format(input_index))

            points_dset = mesh_group.create_dataset('points', data=points, dtype=np.float)

            points_dset.attrs['origin'] = target_indices[input_index]

            mesh_group.create_dataset('triangles', data=triangles, dtype=np.uintp)
            mesh_group.create_dataset('vasculature_vertices', data=global_ref_idx, dtype=np.uintp)


    """
    if len(triangles) > 0:

        filepath = os.path.join(output_directory, 'endothelium_{}.stl'.format(index))

        tris = np.asarray(triangles)

        L.info('Writing {}'.format(filepath))

        cell_mesh = stl.mesh.Mesh(np.zeros(len(triangles), dtype=stl.mesh.Mesh.dtype))

        cell_mesh.vectors = mesh_points[tris]

        cell_mesh.save(filepath)

    else:

        L.info('Empty {}'.format(filepath))
        open(filepath, 'a').close()

    """
