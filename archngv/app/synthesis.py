"""
Synthesize astrocyte morphologies
"""
from pathlib import Path

import click
from archngv.building.morphology_synthesis.data_extraction import astrocyte_circuit_data


class Worker:
    """Morphology synthesis helper"""
    def __init__(self, config, kwargs):
        self._config = config
        self._kwargs = kwargs

    def __call__(self, astrocyte_index):
        import numpy as np
        from archngv.building.morphology_synthesis.full_astrocyte import \
            synthesize_astrocyte

        seed = hash((self._kwargs['seed'], astrocyte_index)) % (2 ** 32)
        np.random.seed(seed)

        paths = _synthesis_input_paths(self._kwargs)
        morph = synthesize_astrocyte(astrocyte_index, paths, self._config)

        cell_properties = astrocyte_circuit_data(astrocyte_index, paths)[0]
        morph.write(Path(paths.morphology_directory, cell_properties.name[0] + '.h5'))


def _synthesis_input_paths(kwargs):
    from archngv.building.morphology_synthesis.data_structures import \
        SynthesisInputPaths
    return SynthesisInputPaths(
            astrocytes=kwargs['astrocytes'],
            microdomains=kwargs['microdomains'],
            neuronal_connectivity=kwargs['neuronal_connectivity'],
            gliovascular_connectivity=kwargs['gliovascular_connectivity'],
            neuroglial_connectivity=kwargs['neuroglial_connectivity'],
            endfeet_areas=kwargs['endfeet_areas'],
            tns_parameters=kwargs['tns_parameters'],
            tns_distributions=kwargs['tns_distributions'],
            tns_context=kwargs['tns_context'],
            morphology_directory=kwargs['out_morph_dir'])


@click.command(help=__doc__)
@click.option("--config", help="Path to synthesis YAML config", required=True)
@click.option("--tns-distributions", help="Path to TNS distributions (JSON)", required=True)
@click.option("--tns-parameters", help="Path to TNS parameters (JSON)", required=True)
@click.option("--tns-context", help="Path to TNS context (JSON)", required=True)
@click.option("--astrocytes", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--gliovascular-connectivity", help="Path to gliovascular connectivity sonata", required=True)
@click.option("--neuroglial-connectivity", help="Path to neuroglial connectivity (HDF5)", required=True)
@click.option("--endfeet-areas", help="Path to HDF5 endfeet areas", required=True)
@click.option("--neuronal-connectivity", help="Path to HDF5 with synapse positions", required=True)
@click.option("--out-morph-dir", help="Path to output morphology folder", required=True)
@click.option("--parallel", help="Parallelize with 'multiprocessing'", is_flag=True, default=False)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
def cmd(config, **kwargs):
    # pylint: disable=missing-docstring
    from archngv.app.utils import ensure_dir, load_yaml, apply_parallel_function
    from archngv.core.datasets import CellData
    from archngv.app.logger import LOGGER

    config = load_yaml(config)
    ensure_dir(kwargs['out_morph_dir'])

    map_func = apply_parallel_function if kwargs['parallel'] else map

    n_astrocytes = len(CellData(kwargs['astrocytes']))

    worker = Worker(config, kwargs)
    for n, _ in enumerate(map_func(worker, range(n_astrocytes)), start=1):
        if n % 1000 == 0:
            LOGGER.info('Synthesized %d astrocytes', n)
