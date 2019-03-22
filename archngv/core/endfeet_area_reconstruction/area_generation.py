import os
import json
import h5py
import logging
import multiprocessing

import numpy as np

from .detail.endfoot import create_endfoot_from_global_data
from .detail.area_fitting import fit_area
from .detail.fmm_growing import FastMarchingEikonalSolver

from ..data_structures import GliovascularConnectivity
from ..util.parallel import distribute_batches


L = logging.getLogger(__name__)


def _dispatch_data(n_cells,
                   mark_indices,
                   mark_offsets,
                   target_areas,
                   target_thicknesses,
                   ngv_config):

    with GliovascularConnectivity(ngv_config.output_paths('gliovascular_connectivity')) as gv_conn:

        for endfoot_index in range(n_cells):

            vasculature_mesh_vertices = \
            mark_indices[mark_offsets[endfoot_index]: mark_offsets[endfoot_index + 1]]

            endfoot_data = {
                                'index': endfoot_index,
                                'astrocyte_index': gv_conn.endfoot.to_astrocyte(endfoot_index),
                                'vasculature_mesh_vertices': vasculature_mesh_vertices.astype(np.uintp),
                                'config': ngv_config
            }

            if target_areas is not None:
                endfoot_data['target_area'] = target_areas[endfoot_index]

            if target_thicknesses is not None:
                endfoot_data['target_thickness'] = target_thicknesses[endfoot_index]

            yield endfoot_data


def process_endfoot(endfoot_data):

    ngv_config = endfoot_data['config']

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

    filepath = os.path.join(ngv_config.endfeetome_directory,
                            'endfoot_{}_{}.stl'.format(endfoot_data['index'],
                                                       endfoot_data['astrocyte_index']))

    #io.export_endfoot_mesh(endfoot, filepath)

    return endfoot_data['index'], endfoot_data['astrocyte_index'], endfoot.coordinates_array, endfoot.triangle_array

shared_data = {}


def run_fast_marching_method(ngv_config, mesh, endfeet_points):

    L.info('fmm started.')

    max_area = ngv_config.parameters["synthesis"]["endfeet_area_reconstruction"]["max_endfoot_area"] 

    threshold_radius = np.sqrt(max_area / 3.1415)

    solver = FastMarchingEikonalSolver(
                                           mesh, 
                                           endfeet_points.astype(np.float32),
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
                            endfeet_points,
                            ngv_config,
                            area_fitting_data=None,
                            thickness_data=None,
                            parallel=False):

    n_endfeet = len(endfeet_points)

    mark_seeds, mark_indices, mark_offsets, travel_times = \
    run_fast_marching_method(ngv_config, mesh, endfeet_points)

    L.info('Generating groups...')

    mesh_points = mesh.points()
    mesh_triangles = mesh.face_vertex_indices()

    shared_data['mesh_points'] = mesh_points.astype(np.float32)
    shared_data['mesh_triangles'] = mesh_triangles.astype(np.uintp)
    shared_data['vertex_travel_times'] = travel_times.astype(np.float32)

    endfeet_data_it = \
    _dispatch_data(n_endfeet,
                   mark_indices,
                   mark_offsets,
                   area_fitting_data,
                   thickness_data,
                   ngv_config)

    L.info('Processing Endfeet..')

    if parallel:

        n_processes = multiprocessing.cpu_count()

        n_chunks = int(np.ceil(n_endfeet / n_processes))

        pool = multiprocessing.Pool(
                                        #iinitializer=init_shared,
                                        #initargs=(mesh_points, mesh_triangles, travel_times),
                                        processes=n_processes
                                   )

        data = pool.imap_unordered(process_endfoot, endfeet_data_it)

    else:

        #init_hared(mesh_points, mesh_triangles, travel_times)
        data = [process_endfoot(d) for d in endfeet_data_it]


    export_data(ngv_config, data)


def export_data(ngv_config, data):

    filepath = os.path.join(ngv_config.endfeetome_directory, 'endfeetome.h5')

    with h5py.File(filepath, 'w') as fd:

        metadata = fd.create_group('metadata')

        metadata.attrs['object_type'] = 'endfoot_mesh'

        meshes = fd.create_group('objects')

        for index, astrocyte_index, points, triangles in data:

            mesh_group = meshes.create_group('endfoot_{}'.format(index))

            mesh_group.create_dataset('points', data=points)
            mesh_group.create_dataset('triangles', data=triangles)

            mesh_group.attrs['astrocyte_index'] = astrocyte_index

            L.info('written {}'.format(index))
