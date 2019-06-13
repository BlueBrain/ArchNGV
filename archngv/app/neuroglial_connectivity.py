"""
Generate neuroglial (N-G) connectivity
"""

import click


@click.command(help=__doc__)
@click.option("--astrocytes", help="Path to astrocyte node population (SONATA Nodes)", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--synaptic-data", help="Path to synaptic data (SONATA Edges HDF5)", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--output-connectivity", help="Path to output HDF5 (connectivity)", required=True)
def cmd(astrocytes, microdomains, synaptic_data, seed, output_connectivity):
    # pylint: disable=missing-docstring,redefined-argument-from-local,too-many-locals
    import numpy as np

    from voxcell.sonata import NodePopulation

    from archngv.core.data_structures.data_microdomains import MicrodomainTesselation
    from archngv.core.data_structures.data_synaptic import SynapticData
    from archngv.core.connectivity.neuroglial_generation import generate_neuroglial
    from archngv.core.exporters.export_neuroglial_connectivity import export_neuroglial_connectivity

    from archngv.app.logger import LOGGER

    astrocytes = NodePopulation.load(astrocytes)

    np.random.seed(seed)

    LOGGER.info('Generating neuroglial connectivity...')

    with MicrodomainTesselation(microdomains) as microdomains, \
         SynapticData(synaptic_data) as syn_data:

        data_iterator = generate_neuroglial(
            astrocytes=astrocytes,
            microdomains=microdomains,
            synaptic_data=syn_data
        )

        LOGGER.info('Exporting the per astrocyte files...')
        export_neuroglial_connectivity(
            data_iterator,
            n_unique_astrocytes=astrocytes.size,
            n_unique_synapses=syn_data.n_synapses,
            n_unique_neurons=syn_data.n_neurons,
            neuroglial_connectivity_filepath=output_connectivity
        )

        LOGGER.info("Done!")
