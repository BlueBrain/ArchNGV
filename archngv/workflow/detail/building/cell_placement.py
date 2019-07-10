import logging
from archngv.workflow.detail.building import helpers
from voxcell import VoxelData

import numpy as np

from archngv.core.data_structures.vasculature_morphology import Vasculature
from archngv.core.cell_placement.positions import create_positions
from archngv.core.exporters import export_cell_placement_data

L = logging.getLogger(__name__)


def create_cell_positions(ngv_config, run_parallel):
    """
    Given the config, generate positions and radii
    of astrocytes.

    Args:
        ngv_config (NGVConfig):
            Contains the parameters, input and output absolute paths that
            are required for the circuit generation.

    Returns:
        positions (2D array)
        radiis (1D array)

    """


    cell_placement_parameters = ngv_config.parameters['cell_placement']

    voxelized_intensity = VoxelData.load_nrrd(ngv_config.input_paths('voxelized_intensity'))
    voxelized_bnregions = VoxelData.load_nrrd(ngv_config.input_paths('voxelized_brain_regions'))

    voxelized_intensity.raw = voxelized_intensity.raw.astype(np.float)

    index_list = []

    vasculature_path = ngv_config.input_paths('vasculature')
    if vasculature_path is not None:
        index_list.append(Vasculature.load(vasculature_path).spatial_index())

    # astro somata pos and radii
    somata_positions, somata_radii = create_positions(cell_placement_parameters,
                                                      voxelized_intensity,
                                                      voxelized_bnregions,
                                                      spatial_indexes=index_list)

    cell_names = np.array(['GLIA_{:013d}'.format(index) for index in range(len(somata_positions))], dtype=bytes)
    cell_ids =  range(len(cell_names))

    export_cell_placement_data(ngv_config.output_paths('cell_data'), cell_ids, cell_names, somata_positions, somata_radii)

    astro_collection = helpers.create_astrocyte_collection(ngv_config, cell_names, somata_positions, somata_radii)

    L.info('Saving mvd3 circuit')

    # save the circuit to file
    astro_collection.save_mvd3(ngv_config.output_paths('circuit'))

"""
if __name__ == '__main__':
    import sys
    from archngv import NGVConfig

    config_path = sys.argv[1]

    config = NGVConfig.from_file(config_path)

    keys = ['vasculature_index']
    if config._config['use_somata_geometry']:
        keys.append('neuronal_somata_index')

    with OnSSDs(config, keys) as _:
        create_cell_positions(config, False)
"""
