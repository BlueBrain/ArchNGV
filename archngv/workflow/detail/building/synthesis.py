import os
import h5py
import numpy as np
import logging
from functools import partial

from archngv.core.morphology_synthesis.full_astrocyte import synthesize_astrocyte

from archngv.core.data_structures.data_cells import CellData
from archngv.core.data_structures.data_ngv import NGVData
from archngv.core.data_structures.connectivity_ngv import NGVConnectome


L = logging.getLogger(__name__)


def apply_func(func, data_generator):
    for data in data_generator:
        func(data)


class Worker(object):

    def __init__(self, ngv_config):

        self._cfg = ngv_config

    def __call__(self, astrocyte_index):

        ngv_config = self._cfg

        tns_parameters_path = ngv_config.input_paths('tns_astrocyte_parameters')
        tns_distributions_path = ngv_config.input_paths('tns_astrocyte_distributions')

        # These are archngv synthesis parameters
        synthesis_parameters = ngv_config.parameters['synthesis']

        cell_data_filepath = ngv_config.output_paths('cell_data')
        microdomains_filepath = ngv_config.output_paths('overlapping_microdomain_structure')
        synaptic_data_filepath = ngv_config.output_paths('synaptic_data')
        gliovascular_data_filepath = ngv_config.output_paths('gliovascular_data')
        gliovascular_conn_filepath = ngv_config.output_paths('gliovascular_connectivity')
        neuroglial_connectivity_filepath = ngv_config.output_paths('neuroglial_connectivity')
        parameters = ngv_config.parameters['synthesis']

        if 'seed' in ngv_config._config:
            seed = hash((ngv_config._config['seed'], astrocyte_index)) % (2 ** 32)
            np.random.seed(seed)

        synthesize_astrocyte(astrocyte_index,
                             cell_data_filepath,
                             microdomains_filepath,
                             synaptic_data_filepath,
                             gliovascular_data_filepath,
                             gliovascular_conn_filepath,
                             neuroglial_connectivity_filepath,
                             tns_parameters_path,
                             tns_distributions_path,
                             ngv_config.morphology_directory,
                             parameters)


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

        astrocyte_ids = cell_data.astrocyte_gids[:]

    worker = Worker(ngv_config)

    map_func(worker, astrocyte_ids)
