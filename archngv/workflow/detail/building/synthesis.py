import h5py
import logging

from archngv.core.morphology_synthesis.full_astrocyte import synthesize_astrocytes
from archngv.core.morphology_synthesis.projections import synthesize_astrocyte_endfeet

from archngv.core.data_structures.data_cells import CellData


L = logging.getLogger(__name__)


def apply_func(func, data_generator):
    for data in data_generator:
        func(data)

def apply_parallel_func(func, data_generator):

    import multiprocessing

    n_cores = multiprocessing.cpu_count()

    L.info('Run in parallel enabled. N cores: {}'.format(n_cores))

    with multiprocessing.Pool(n_cores) as p:

        for _ in p.imap_unordered(func, data_generator):
            pass


def create_synthesized_morphologies(ngv_config, run_parallel):

    map_func = apply_parallel_func if run_parallel else apply_func

    with CellData(ngv_config.output_paths('cell_data')) as cell_data:

        somata_positions = cell_data.astrocyte_positions
        cell_names = cell_data.astrocyte_names

        synthesize_astrocytes(ngv_config, somata_positions, cell_names, map_func)


def create_endfeet_morphologies(ngv_config, run_parallel):

    map_func = apply_parallel_func if run_parallel else apply_func

    with CellData(ngv_config.output_paths('cell_data')) as cell_data:

        cell_ids = cell_data.astrocyte_gids[:]

    synthesize_astrocyte_endfeet(ngv_config, cell_ids, map_func)
