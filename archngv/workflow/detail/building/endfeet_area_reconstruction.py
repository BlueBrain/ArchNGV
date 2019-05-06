import h5py
import logging
import numpy as np
from scipy import stats

import openmesh

from archngv.core.data_structures.data_cells import CellData
from archngv.core.data_structures.data_gliovascular import GliovascularData
from archngv.core.data_structures.connectivity_gliovascular import GliovascularConnectivity

from archngv.core.endfeet_area_reconstruction.area_generation import endfeet_area_generation
from archngv.core.endfeet_area_reconstruction.area_generation import endfeet_area_extraction

from archngv.core.exporters.export_endfeet_areas import export_endfeet_areas

L = logging.getLogger(__name__)


def create_endfeet_areas(ngv_config, run_parallel):

    L.info('Endfeet Area Generation started.')
    parameters = ngv_config.parameters["synthesis"]["endfeet_area_reconstruction"]

    L.info('Loading vasculature mesh...')
    vasculature_mesh = openmesh.read_trimesh(ngv_config.input_paths('vasculature_mesh'))

    gliovascular_data_path = ngv_config.output_paths('gliovascular_data')
    gliovascular_connectivity_path = ngv_config.output_paths('gliovascular_connectivity')


    data_generator = endfeet_area_generation(vasculature_mesh,
                                             parameters,
                                             gliovascular_data_path,
                                             gliovascular_connectivity_path,
                                             parallel=run_parallel)

    output_path = ngv_config.output_paths('endfeetome')
    export_endfeet_areas(output_path, data_generator)

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

