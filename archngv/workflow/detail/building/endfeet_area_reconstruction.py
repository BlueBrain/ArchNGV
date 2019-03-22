import h5py
import logging
import numpy as np
from scipy import stats

import openmesh

from archngv.core.data_structures.data_cells import CellData
from archngv.core.data_structures.data_gliovascular import GliovascularData
from archngv.core.data_structures.connectivity_gliovascular import GliovascularConnectivity

from archngv.core.endfeet_area_reconstruction.area_generation import endfeet_area_generation


L = logging.getLogger(__name__)


def endfeet_area_extraction(area_distribution_dict, n_endfeet):

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


def create_endfeet_areas(ngv_config, run_parallel):

    L.info('Endfeet Area Generation started.')

    parameters = ngv_config.parameters["synthesis"]["endfeet_area_reconstruction"]

    L.info('Loading vasculature mesh...')

    mesh = openmesh.read_trimesh(ngv_config.input_paths('vasculature_mesh'))

    with GliovascularData(ngv_config.output_paths('gliovascular_data')) as gdata:
        surface_endfeet_targets = gdata.endfoot_surface_coordinates[:]

    L.info('Endfeet area algorithm started...')
    endfeet_target_areas = endfeet_area_extraction(parameters["area_constraints"], len(surface_endfeet_targets))

    endfeet_area_generation(mesh,
                            surface_endfeet_targets,
                            ngv_config,
                            area_fitting_data=endfeet_target_areas,
                            thickness_data=None,
                            parallel=run_parallel)

    L.info('Endfeet Area Generation completed.')

    try:
        joined_mesh_filename = ngv_config.output_paths('joined_mesh_filename')
    except KeyError:
        joined_mesh_filename=None


if __name__ == '__main__':
    import sys
    from archngv import NGVConfig

    config = NGVConfig.from_file(sys.argv[1])
    create_endfeet_areas(config, False)

