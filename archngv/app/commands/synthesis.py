"""
Synthesize astrocyte morphologies
"""
from pathlib import Path
import time

import click
from dask import bag
from dask.distributed import Client, progress
import dask_mpi

from archngv.app.utils import load_yaml, random_generator


def _synthesize(astrocyte_index, seed, paths, config):
    # imports must be local, otherwise when used with modules, they use numpy of the loaded
    # module which might be outdated
    from archngv.building.morphology_synthesis.data_extraction import astrocyte_circuit_data
    from archngv.building.morphology_synthesis.full_astrocyte import synthesize_astrocyte

    seed = hash((seed, astrocyte_index)) % (2 ** 32)
    rng = random_generator(seed)

    morph = synthesize_astrocyte(astrocyte_index, paths, config, rng)
    cell_properties = astrocyte_circuit_data(astrocyte_index, paths, rng)[0]
    morph.write(Path(paths.morphology_directory, cell_properties.name[0] + '.h5'))


@click.command(help=__doc__)
@click.option("--config", help="Path to synthesis YAML config", required=True)
@click.option("--tns-distributions", help="Path to TNS distributions (JSON)", required=True)
@click.option("--tns-parameters", help="Path to TNS parameters (JSON)", required=True)
@click.option("--tns-context", help="Path to TNS context (JSON)", required=True)
@click.option("--astrocytes", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option(
    "--gliovascular-connectivity", help="Path to gliovascular connectivity sonata", required=True)
@click.option(
    "--neuroglial-connectivity", help="Path to neuroglial connectivity (HDF5)", required=True)
@click.option("--endfeet-areas", help="Path to HDF5 endfeet areas", required=True)
@click.option("--neuronal-connectivity", help="Path to HDF5 with synapse positions", required=True)
@click.option("--out-morph-dir", help="Path to output morphology folder", required=True)
@click.option("--parallel", help="Use Dask's mpi client", is_flag=True, default=False)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
def cmd(config,
        tns_distributions,
        tns_parameters,
        tns_context,
        astrocytes,
        microdomains,
        gliovascular_connectivity,
        neuroglial_connectivity,
        endfeet_areas,
        neuronal_connectivity,
        out_morph_dir,
        parallel,
        seed):
    # pylint: disable=too-many-arguments
    """Cli interface to synthesis."""
    from archngv.core.datasets import CellData
    from archngv.building.morphology_synthesis.data_structures import SynthesisInputPaths

    if parallel:
        dask_mpi.initialize()
        client = Client()
    else:
        client = Client(processes=False, threads_per_worker=1)

    Path(out_morph_dir).mkdir(exist_ok=True, parents=True)
    config = load_yaml(config)
    n_astrocytes = len(CellData(astrocytes))
    paths = SynthesisInputPaths(
        astrocytes=astrocytes,
        microdomains=microdomains,
        neuronal_connectivity=neuronal_connectivity,
        gliovascular_connectivity=gliovascular_connectivity,
        neuroglial_connectivity=neuroglial_connectivity,
        endfeet_areas=endfeet_areas,
        tns_parameters=tns_parameters,
        tns_distributions=tns_distributions,
        tns_context=tns_context,
        morphology_directory=out_morph_dir)

    synthesize = bag.from_sequence(range(n_astrocytes), partition_size=1) \
        .map(_synthesize, seed=seed, paths=paths, config=config) \
        .persist()
    # print is intentional here because it is for showing the progress bar title
    print(f'Synthesizing {n_astrocytes} astrocytes')
    progress(synthesize)
    synthesize.compute()

    time.sleep(10)  # this sleep is necessary to let dask syncronize state across the cluster
    client.retire_workers()
