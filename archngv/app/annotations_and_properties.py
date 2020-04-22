"""
Synthesize astrocyte morphologies
"""
from collections import namedtuple
import click


class Worker:
    """Morphology synthesis helper"""
    def __init__(self, config, seed, paths):
        self._config = config
        self._paths = paths
        self._seed = seed

    def __call__(self, astrocyte_index):
        import numpy as np
        from archngv.core.datasets import CellData
        from archngv.building.morphology_synthesis.annotation import create_astrocyte_annotations
        from archngv.building.morphology_synthesis.properties import create_astrocyte_properties

        seed = hash((self._seed, astrocyte_index)) % (2 ** 32)
        np.random.seed(seed)

        cell_data = CellData(self._paths.cell_data)
        cell_name = str(cell_data.astrocyte_names[astrocyte_index], 'utf-8')

        return (cell_name,
                create_astrocyte_annotations(astrocyte_index, self._paths),
                create_astrocyte_properties(astrocyte_index, self._paths))


def _apply_func(func, data_generator):
    return map(func, data_generator)


def _apply_parallel_func(func, data_generator):
    import multiprocessing
    from archngv.app.logger import LOGGER
    n_cores = multiprocessing.cpu_count()
    LOGGER.info('Run in parallel enabled. N cores: %d', n_cores)
    with multiprocessing.Pool(n_cores) as p:
        for result in p.imap_unordered(func, data_generator):
            yield result


Paths = namedtuple('Paths', ['cell_data',
                            'synaptic_data',
                            'gliovascular_data',
                            'gliovascular_connectivity',
                            'neuroglial_connectivity',
                            'endfeet_areas',
                            'morphology_directory'])


@click.command(help=__doc__)
@click.option("--config", help="Path to synthesis YAML config", required=True)
@click.option("--cell-data", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--gliovascular-connectivity", help="Path to gliovascular connectivity (HDF5)", required=True)
@click.option("--gliovascular-data", help="Path to gliovascular data (HDF5)", required=True)
@click.option("--neuroglial-connectivity", help="Path to neuroglial connectivity (HDF5)", required=True)
@click.option("--synaptic-data", help="Path to HDF5 with synapse positions", required=True)
@click.option("--endfeet-areas", help="Path to HDF5 endfeet areas", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--parallel", help="Parallelize with 'multiprocessing'", is_flag=True, default=False)
@click.option("--morph-dir", help="Path to morphology folder", required=True)
@click.option("--annotations_output", help="Path to output HDF5", required=True)
@click.option("--properties_output", help="Path to output HDF5", required=True)
def cmd(config, **kwargs):
    # pylint: disable=missing-docstring
    from archngv.core.datasets import CellData
    from archngv.app.utils import load_yaml
    from archngv.building.exporters.export_ngv_data import export_annotations_and_properties

    config = load_yaml(config)

    paths = Paths(
        cell_data=kwargs['cell_data'],
        synaptic_data=kwargs['synaptic_data'],
        gliovascular_data=kwargs['gliovascular_data'],
        gliovascular_connectivity=kwargs['gliovascular_connectivity'],
        neuroglial_connectivity=kwargs['neuroglial_connectivity'],
        endfeet_areas=kwargs['endfeet_areas'],
        morphology_directory=kwargs['morph_dir']
    )

    map_func = _apply_parallel_func if kwargs['parallel'] else _apply_func

    with CellData(kwargs['cell_data']) as cell_data:
        astrocyte_ids = cell_data.astrocyte_gids[:]

    worker = Worker(config, kwargs['seed'], paths)
    annotations_and_properties_iterable = map_func(worker, astrocyte_ids)
    export_annotations_and_properties(kwargs['annotations_output'],
                                      kwargs['properties_output'],
                                      annotations_and_properties_iterable)
