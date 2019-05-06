import os
import json
import h5py
import logging
import multiprocessing

import numpy as np
from scipy import stats

from .detail.endfoot import create_endfoot_from_global_data
from .detail.area_fitting import fit_area
from .detail.fmm_growing import FastMarchingEikonalSolver

from ..data_structures import GliovascularData
from ..data_structures import GliovascularConnectivity
from ..util.parallel import distribute_batches


L = logging.getLogger(__name__)


def _dispatch_data(n_cells,
                   mark_indices,
                   mark_offsets,
                   target_areas,
                   target_thicknesses,
                   gliovascular_connectivity_path):
    """ Dispatch the data for one astrocyte endfoot

    Args:
        mark_indices: int[N, 1]
            Vasculature indices for all endfoot mesh vertices.
        mark_offsets: int[M + 1, 1]
            The vasculature indices of the i-th endfoot (total M) can be retrieved as
            mark_indices[mark_offsets[i]: mark_offsets[i + 1]]
        target_areas: float[M, 1]
            Target area for each endfoot mesh.
        target_thicknesses: float[M, 1]
            Target thickness for each endfoot mesh (Not implemented)
        gliovascular_connectivity_path: string
            Path to gliovascular connectivity.

    Returns: dict
    """
    with GliovascularConnectivity(gliovascular_connectivity_path) as gv_conn:

        for endfoot_index in range(n_cells):

            vasculature_mesh_vertices = \
            mark_indices[mark_offsets[endfoot_index]: mark_offsets[endfoot_index + 1]]

            endfoot_data = {
                                'index': endfoot_index,
                                'astrocyte_index': gv_conn.endfoot.to_astrocyte(endfoot_index),
                                'vasculature_mesh_vertices': vasculature_mesh_vertices.astype(np.uintp)
            }

            if target_areas is not None:
                endfoot_data['target_area'] = target_areas[endfoot_index]

            if target_thicknesses is not None:
                endfoot_data['target_thickness'] = target_thicknesses[endfoot_index]

            yield endfoot_data


def process_endfoot(endfoot_data):
    """ Shrink endfoot area if needed.
    """
    mesh_coordinates = shared_data['mesh_points']
    mesh_triangles = shared_data['mesh_triangles']
    travel_times = shared_data['vertex_travel_times']

    vasculature_mesh_vertices = endfoot_data['vasculature_mesh_vertices']

    endfoot = create_endfoot_from_global_data(endfoot_data['index'],
                                              mesh_coordinates,
                                              mesh_triangles,
                                              vasculature_mesh_vertices)

    if endfoot.number_of_triangles > 3:

        if 'target_area' in endfoot_data:

            endfoot.extra = \
            {'vertex': {'travel_times': travel_times[endfoot.vasculature_vertices]}}

            #L.info('Endfeet area fiting data provided.')
            fit_area(endfoot, endfoot_data['target_area'])


        if 'target_thickness' in endfoot_data:
            raise NotImplementedError

    return endfoot_data['index'], endfoot.coordinates_array, endfoot.triangle_array


shared_data = {}


def run_fast_marching_method(mesh, endfeet_points, max_area):
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
    L.info('fmm started.')


    threshold_radius = np.sqrt(max_area / 3.1415)

    solver = FastMarchingEikonalSolver(
                                           mesh, 
                                           endfeet_points,
                                           threshold_radius
                                      )
    solver.solve()

    L.info('Extract travel times..')
    travel_times = solver.travel_times()

    L.info('Extract groups..')
    mark_seeds, mark_indices, mark_offsets = solver.groups()

    L.info('fmm completed')
    return mark_seeds, mark_indices, mark_offsets, travel_times


def endfeet_area_generation(mesh,
                            parameters,
                            gliovascular_data_path,
                            gliovascular_connectivity_path,
                            parallel=False):
    """ Generate endfeet areas

    Args:
        mesh: Trimesh
            The mesh of the vasculature
        parameters: dict
            The parameters for the algorithms
        gliovascular_data_path: string
            Path to gliovascular data h5
        gliovascular_connectivity_path: string
            Path to gliovascular connectivity h5
        parallel: bool
            Enable parallel run for the endfeet areas
    """

    with GliovascularData(gliovascular_data_path) as gdata:
        endfeet_points = gdata.endfoot_surface_coordinates[:].astype(np.float32)

    n_endfeet = len(endfeet_points)

    mark_seeds, mark_indices, mark_offsets, travel_times = \
        run_fast_marching_method(mesh, endfeet_points, parameters['max_endfoot_area'])

    L.info('Generating groups...')

    mesh_points = mesh.points()
    mesh_triangles = mesh.face_vertex_indices()

    shared_data['mesh_points'] = mesh_points.astype(np.float32)
    shared_data['mesh_triangles'] = mesh_triangles.astype(np.uintp)
    shared_data['vertex_travel_times'] = travel_times.astype(np.float32)

    if 'area_constraints' in parameters:
        endfeet_target_areas = \
            endfeet_area_extraction(parameters["area_constraints"], n_endfeet)
    else:
        endfeet_target_areas = None

    if 'thickness_constraints' in parameters:
        raise NotImplementedError
    else:
        endfeet_target_thicknesses = None

    endfeet_data_it = _dispatch_data(n_endfeet,
                                     mark_indices,
                                     mark_offsets,
                                     endfeet_target_areas,
                                     endfeet_target_thicknesses,
                                     gliovascular_connectivity_path)

    L.info('Processing Endfeet..')

    if parallel:

        n_processes = multiprocessing.cpu_count()

        n_chunks = int(np.ceil(n_endfeet / n_processes))

        pool = multiprocessing.Pool(
                                        #iinitializer=init_shared,
                                        #initargs=(mesh_points, mesh_triangles, travel_times),
                                        processes=n_processes
                                   )

        return pool.imap_unordered(process_endfoot, endfeet_data_it)

    return [process_endfoot(d) for d in endfeet_data_it]


def endfeet_area_extraction(area_distribution_dict, n_endfeet):
    """ Create endfeet area distribution or get values from config
    """
    entry_type = area_distribution_dict['type']
    entry_data = area_distribution_dict['values']

    if entry_type == 'number':

        endfeet_areas = np.ones(n_endfeet) * entry_data
        L.info('Area distribution entry type is number. Broadcastng..')

    elif entry_type == 'list':

        endfeet_areas = np.asarray(map(float, entry_data), dtype=np.float)
        L.info('Area contraints entry is list. Using as is..')

    elif entry_type == 'distribution':

        endfeet_areas = getattr(stats, entry_data[0])(*(float(entry_data[1]), float(entry_data[2]))).rvs(n_endfeet)
        L.info('Area constraints entry is a distribution. Sampling...')

    else:

        raise TypeError("Area constraints type is unknown")

    return endfeet_areas
