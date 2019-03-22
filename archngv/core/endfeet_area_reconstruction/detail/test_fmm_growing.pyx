import logging
from .fmm_growing import FastMarchingEikonalSolver as FMM1
from .mesh_operations cimport find_surface_contours

from scipy import sparse
import numpy as np
import trimesh
import openmesh
from collections import deque

from .area_fitting import fit_area

from scipy.spatial import cKDTree

L = logging.getLogger(__name__)

from .endfoot import create_endfoot_from_global_data


def sample_spherical(npoints, ndim=3):
    """ random points on unit sphere
    """
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

cpdef test_endfeet_area_generation_bak1(filename):

    np.random.seed(5)

    # TODO: find out to load mesh without to process it. Trimesh seems
    # to not generate mesh info if process is False.
    #mesh = trimesh.load(filename, process=True, validate=False)

    n_sources = 5

    mesh = openmesh.read_trimesh(filename)

    endfeet_points = sample_spherical(n_sources).T + mesh.points().mean(axis=0)

    solver = FMM1(mesh, endfeet_points.astype(np.float32), 1.0)
    solver.solve()

    mark_seeds, mark_indices, mark_offsets = solver.groups()
    travel_times = solver.travel_times()

    coordinates = mesh.points()
    triangles = mesh.face_vertex_indices().astype(np.intp)

    endfeet = []

    for i, (off1, off2) in enumerate(zip(mark_offsets[:-1], mark_offsets[1:])):

        surf_idx = mark_indices[off1: off2]

        endfoot = create_endfoot_from_global_data(np.uintp(i), coordinates.astype(np.float32), triangles.astype(np.uintp), surf_idx.astype(np.uintp))

        endfoot.extra = {}

        contour = [endfoot.local_to_global_map[local_vertex] for local_vertex in find_surface_contours(endfoot.edge_to_triangles)]

        endfoot.extra['vertex'] = {'travel_times': travel_times[endfoot.vasculature_vertices]}
        endfoot.extra['target_area'] = 0.5
        fit_area(endfoot, endfoot.extra['target_area'])


        endfeet.append([endfoot, endfoot.extra['target_area'], contour])

    return solver, mesh, endfeet

def main(filename):
    viz_result(*test_endfeet_area_generation_bak1(filename))

cpdef viz_result(solver, open_mesh, endfeet):

    import trimesh
    n_vertices = open_mesh.n_vertices()


    travel_times = solver.travel_times()

    marks = solver.marks()

    unique_marks = np.unique(marks)

    num_colors = len(unique_marks)

    colors = np.asarray([trimesh.visual.random_color() for _ in range(num_colors)])

    vertex_colors = 255 * np.ones([n_vertices, 4], dtype=np.uintp)
    """
    for index, mark in enumerate(unique_marks):
        if mark != -1:

            mask = marks == mark

            vertex_colors[mask] = colors[index]
    """
    for i, (endfoot, target_area, contour) in enumerate(endfeet):

        print('endfoot area: {}, target area: {}'.format(endfoot.area, target_area))

        verts = endfoot.vasculature_vertices

        vertex_colors[verts] = colors[i]

        contour_arr = np.asarray(contour)

        print("contour: {}".format(contour_arr))

        vertex_colors[contour_arr] = (0, 0, 0, 255)

    """
    mask = ~np.isinf(travel_times)
    all_values = travel_times

    max_value = all_values[mask].max()

    num_colors = 10

    dval = max_value / float(num_colors)

    colors = [trimesh.visual.random_color() for _ in xrange(num_colors)]

    for index, value in enumerate(travel_times):

        if not np.isinf(value):
            color_index = min((int(value / dval), num_colors - 1))

            vertex_colors[index][0] = colors[color_index][0]
    """
    mesh = trimesh.Trimesh(vertices=open_mesh.points(),
                           faces=open_mesh.face_vertex_indices(),
                           vertex_colors=vertex_colors)


    mesh.show()



def _it(mark_indices, mark_offsets):

    yield (0, mark_indices[0: mark_offsets[0]])

    for i in range(1, len(mark_offsets)):
        yield (i, mark_indices[mark_offsets[i - 1]: mark_offsets[i]])


def test_endfeet_reconstruction(solver, mesh):
    from archngv.synthesis.endfeet_reconstruction.endfoot import Endfoot
    from archngv.synthesis.endfeet_reconstruction.area_fitting import fit_area
