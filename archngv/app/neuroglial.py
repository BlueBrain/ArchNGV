"""
Generate neuroglial (N-G) connectivity
"""

import click


@click.group(help=__doc__)
def group():
    # pylint: disable=missing-docstring
    pass


@group.command()
@click.option("--neurons", help="Path to neuron node population (SONATA Nodes)", required=True)
@click.option("--astrocytes", help="Path to astrocyte node population (SONATA Nodes)", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--synaptic-data", help="Path to synaptic data (SONATA Edges HDF5)", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output file (SONATA Edges HDF5)", required=True)
def connectivity(neurons, astrocytes, microdomains, synaptic_data, seed, output):
    """ Generate N-G connectivity """
    # pylint: disable=redefined-argument-from-local,too-many-locals
    import numpy as np

    from voxcell.sonata import NodePopulation

    from archngv import MicrodomainTesselation, SynapticData
    from archngv.building.connectivity.neuroglial_generation import generate_neuroglial
    from archngv.building.exporters.export_neuroglial_connectivity import export_neuroglial_connectivity

    from archngv.app.logger import LOGGER

    neurons = NodePopulation.load(neurons)
    astrocytes = NodePopulation.load(astrocytes)

    np.random.seed(seed)

    LOGGER.info('Generating neuroglial connectivity...')

    microdomains = MicrodomainTesselation(microdomains)

    with SynapticData(synaptic_data) as syn_data:

        data_iterator = generate_neuroglial(
            astrocytes=astrocytes,
            microdomains=microdomains,
            synaptic_data=syn_data
        )

        LOGGER.info('Exporting the per astrocyte files...')
        export_neuroglial_connectivity(
            data_iterator,
            neurons=neurons,
            astrocytes=astrocytes,
            output_path=output
        )

        LOGGER.info("Done!")


@group.command()
@click.option("-i", "--input", help="Path to input file (SONATA Edges HDF5)", required=True)
@click.option("--astrocytes", help="Path to astrocyte node population (SONATA Nodes)", required=True)
@click.option("--annotations", help="Path file with synapse annotations", required=True)
@click.option("-o", "--output", help="Path to output file (SONATA Edges HDF5)", required=True)
def bind_annotations(input, astrocytes, annotations, output):  # pylint: disable=redefined-builtin
    """ Bind synapse annotations with closest astrocyte sections. """
    import shutil

    from voxcell.sonata import NodePopulation

    from archngv.building.exporters.export_neuroglial_connectivity import bind_annotations as _run

    astrocytes = NodePopulation.load(astrocytes)

    shutil.copyfile(input, output)
    _run(output, astrocytes=astrocytes, annotations_file=annotations)
