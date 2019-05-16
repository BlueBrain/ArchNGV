import h5py
import logging
import numpy as np

from voxcell import VoxelData
from archngv.core.vasculature_morphology import Vasculature
from archngv.core.connectivity.gliovascular_generation import generate_gliovascular
from archngv.core.data_structures.data_cells import CellData
from archngv.core.data_structures.data_microdomains import MicrodomainTesselation
from archngv.core.exporters import export_gliovascular_data
from archngv.core.exporters import export_gliovascular_connectivity

from . import helpers

L = logging.getLogger(__name__)


def create_gliovascular_connectivity(ngv_config, run_parallel):

    assert run_parallel == False

    params = ngv_config.parameters['gliovascular_connectivity']

    vasculature_path = ngv_config.input_paths('vasculature')

    L.info('Loading vasculature skeleton {}'.format(vasculature_path))

    vasculature = Vasculature.load(vasculature_path)

    with \
        CellData(ngv_config.output_paths('cell_data')) as data, \
        MicrodomainTesselation(ngv_config.output_paths('overlapping_microdomain_structure')) as microdomains:

        somata_positions = data.astrocyte_positions[:]

        n_astrocytes = len(somata_positions)

        somata_idx = np.arange(len(somata_positions), dtype=np.uintp)

        endfeet_surface_positions, \
        endfeet_graph_positions, \
        endfeet_to_astrocyte_mapping, \
        endfeet_to_vasculature_mapping = \
        generate_gliovascular(somata_idx, somata_positions, microdomains, vasculature, params)

    L.info('Exporting gliovascular data...')

    export_gliovascular_data(
                                ngv_config.output_paths('gliovascular_data'),
                                endfeet_surface_positions,
                                endfeet_graph_positions
                            )

    L.info('Exporting gliovascular connectivity...')

    export_gliovascular_connectivity(
                                        ngv_config.output_paths('gliovascular_connectivity'),
                                        n_astrocytes,
                                        endfeet_to_astrocyte_mapping,
                                        endfeet_to_vasculature_mapping
                                    )

