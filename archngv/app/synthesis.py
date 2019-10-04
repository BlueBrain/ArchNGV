"""
Synthesize astrocyte morphologies
"""

import click


class Worker(object):
    """Morphology synthesis helper"""
    def __init__(self, config, kwargs):
        self._config = config
        self._kwargs = kwargs

    def __call__(self, astrocyte_index):
        import numpy as np

        from archngv.building.morphology_synthesis.full_astrocyte import synthesize_astrocyte

        seed = hash((self._kwargs['seed'], astrocyte_index)) % (2 ** 32)
        np.random.seed(seed)

        synthesize_astrocyte(
            astrocyte_index,
            cell_data_path=self._kwargs['cell_data'],
            microdomains_path=self._kwargs['microdomains'],
            synaptic_data_path=self._kwargs['synaptic_data'],
            gliovascular_data_path=self._kwargs['gliovascular_data'],
            gliovascular_connectivity_path=self._kwargs['gliovascular_connectivity'],
            neuroglial_conn_path=self._kwargs['neuroglial_connectivity'],
            endfeet_areas_path=self._kwargs['endfeet_areas'],
            tns_parameters_path=self._kwargs['tns_parameters'],
            tns_distributions_path=self._kwargs['tns_distributions'],
            morphology_directory=self._kwargs['out_morph_dir'],
            parameters=self._config
        )


def _apply_func(func, data_generator):
    for data in data_generator:
        func(data)


def _apply_parallel_func(func, data_generator):
    import multiprocessing
    from archngv.app.logger import LOGGER
    n_cores = multiprocessing.cpu_count()
    LOGGER.info('Run in parallel enabled. N cores: %d', n_cores)
    with multiprocessing.Pool(n_cores) as p:
        for _ in p.imap_unordered(func, data_generator):
            pass


@click.command(help=__doc__)
@click.option("--config", help="Path to synthesis YAML config", required=True)
@click.option("--tns-distributions", help="Path to TNS distributions (JSON)", required=True)
@click.option("--tns-parameters", help="Path to TNS parameters (JSON)", required=True)
@click.option("--cell-data", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--gliovascular-connectivity", help="Path to gliovascular connectivity (HDF5)", required=True)
@click.option("--gliovascular-data", help="Path to gliovascular data (HDF5)", required=True)
@click.option("--neuroglial-connectivity", help="Path to neuroglial connectivity (HDF5)", required=True)
@click.option("--synaptic-data", help="Path to HDF5 with synapse positions", required=True)
@click.option("--endfeet-areas", help="Path to HDF5 endfeet areas", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--parallel", help="Parallelize with 'multiprocessing'", is_flag=True, default=False)
@click.option("--out-morph-dir", help="Path to output morphology folder", required=True)
def cmd(config, **kwargs):
    # pylint: disable=missing-docstring
    from archngv import CellData
    from archngv.app.utils import load_yaml, ensure_dir

    config = load_yaml(config)

    map_func = _apply_parallel_func if kwargs['parallel'] else _apply_func

    with CellData(kwargs['cell_data']) as cell_data:
        astrocyte_ids = cell_data.astrocyte_gids[:]

    worker = Worker(config, kwargs)

    ensure_dir(kwargs['out_morph_dir'])
    map_func(worker, astrocyte_ids)
