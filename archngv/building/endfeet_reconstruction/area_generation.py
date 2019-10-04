""" Endfeet areas generation processing """

import logging
import multiprocessing

import numpy as np

from archngv import GliovascularData

# pylint: disable=no-name-in-module
from archngv.building.endfeet_reconstruction.detail import area_fitting, fmm_growing
from archngv.building.endfeet_reconstruction.detail.endfoot import create_endfoot_from_global_data


from archngv.utils.statistics import truncated_normal
from archngv.utils.decorators import log_execution_time, log_start_end

L = logging.getLogger(__name__)

# data shared in the multiprocessing child forks. Works only in unixs
# https://stackoverflow.com/questions/27161412/how-does-copy-on-write-work-in-fork/27161457
_SHARED_DATA = {}


def _sample_distribution(params, key, n_samples):
    """ Samples from a truncated normal distribution
    Args:
        params: dict
        key : str
    """
    mean_sdev_min_max = params[key]
    return truncated_normal(*mean_sdev_min_max).rvs(n_samples)


def process_endfoot(endfoot_index):
    """ Processes the endfoot with endfoot_index

    Notes:
        Each endfoot is represented as a collection of vasculature mesh vertices. Thus, we
        first extract these indices via the offsets which tell us how to slice the indices
        mark dataset to extract the indices for this endfoot.

        Then from the endfoot vertices and the vasculature mesh, we extract the submesh triangulation
        which will become a separate entity. Due to the fact that the grown area is usually overestimated
        in the growth algorithm, we shrink the area until we meet with the endfoot area distribution.

        Finally a thickness value is assigned to the astrocyte.
    """
    mark_offs = _SHARED_DATA['mark_offsets']
    mark_inds = _SHARED_DATA['mark_indices']

    mesh_coos = _SHARED_DATA['mesh_points']
    mesh_tris = _SHARED_DATA['mesh_triangles']

    travel_times = _SHARED_DATA['travel_times']

    # vertices on the vasculature that belong to this endfoot
    endfoot_mesh_verts = mark_inds[mark_offs[endfoot_index]: mark_offs[endfoot_index + 1]]
    endfoot = create_endfoot_from_global_data(endfoot_index, mesh_coos, mesh_tris, endfoot_mesh_verts)

    # don't try shrinking the endfoot if it has too few triangles
    if endfoot.number_of_triangles >= 3:

        endfoot.extra = {
            'vertex': {'travel_times': travel_times[endfoot.vasculature_vertices]}}

        # sample a target area distribution and shrink endfeet area
        # to match that value
        endfoot_area = _SHARED_DATA['areas'][endfoot_index]
        area_fitting.fit_area(endfoot, endfoot_area)
    else:
        L.info('Endfoot with index %d has less than 3 triangles', endfoot_index)

    endfoot_thickness = _SHARED_DATA['thicknesses'][endfoot_index]

    L.info('Endfoot with index %d completed.', endfoot_index)
    return endfoot_index, endfoot.coordinates_array, endfoot.triangle_array, endfoot_thickness


@log_start_end
@log_execution_time
def grow_endfeet_areas(vasculature_mesh, endfeet_points, max_area):
    """
    Args:
        mesh: TriMesh
            Vasculature mesh
        endfeet_points: array[float, 3]
            The coordinates of the endfeet contacts on the surface
            of the vasculature.
        max_area: float
            Maximum permitted area for the growht of the endfeet.
    """
    threshold_radius = np.sqrt(max_area / np.pi)

    solver = fmm_growing.FastMarchingEikonalSolver(
        vasculature_mesh,
        endfeet_points,
        threshold_radius
    )
    solver.solve()

    L.info('Extract travel times..')
    travel_times = solver.travel_times()

    L.info('Extract groups..')
    mark_indices, mark_offsets = solver.groups()

    L.info('fmm completed')
    return mark_indices, mark_offsets, travel_times


def endfeet_area_generation(vasculature_mesh,
                            parameters,
                            gliovascular_data_path,
                            parallel):
    """ Generate endfeet areas

    Args:
        vasculature_mesh: Trimesh
            The mesh of the vasculature
        parameters: dict
            The parameters for the algorithms
        gliovascular_data_path: string
            Path to gliovascular data h5
        parallel: bool
            Enable parallel run for the endfeet areas
    """
    with GliovascularData(gliovascular_data_path) as gdata:
        endfeet_points = gdata.endfoot_surface_coordinates[:]

    n_endfeet = len(endfeet_points)

    (_SHARED_DATA['mark_indices'],
     _SHARED_DATA['mark_offsets'],
     _SHARED_DATA['travel_times']) = grow_endfeet_areas(
        vasculature_mesh,
        endfeet_points,
        parameters['area_distribution'][3])

    _SHARED_DATA['mesh_points'] = vasculature_mesh.points()
    _SHARED_DATA['mesh_triangles'] = vasculature_mesh.face_vertex_indices().astype(np.uintp)

    _SHARED_DATA['areas'] = _sample_distribution(parameters, 'area_distribution', n_endfeet)
    _SHARED_DATA['thicknesses'] = _sample_distribution(parameters, 'thickness_distribution', n_endfeet)

    if parallel:
        n_processes = multiprocessing.cpu_count()
        # number of endfeet that will be processed per processor, minimum 1 chunk
        chunk_size = max(1, n_endfeet // n_processes)
        pool = multiprocessing.Pool(processes=n_processes)
        L.info('Parallel started. Number of processes: %d Chunk size: %d', n_processes, chunk_size)
        return pool.imap_unordered(process_endfoot, range(n_endfeet), chunksize=chunk_size)

    L.info('Serial processing started.')
    return map(process_endfoot, range(n_endfeet))
