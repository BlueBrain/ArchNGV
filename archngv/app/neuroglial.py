"""
Generate neuroglial (N-G) connectivity
"""

import click
import numpy as np


@click.group(help=__doc__)
def group():
    # pylint: disable=missing-docstring
    pass


@group.command()
@click.option("--neurons", help="Path to neuron node population (SONATA Nodes)", required=True)
@click.option("--astrocytes", help="Path to astrocyte node population (SONATA Nodes)", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--neuronal-connectivity", help="Path to neuron-neuron sonata edge file", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output file (SONATA Edges HDF5)", required=True)
def connectivity(neurons, astrocytes, microdomains, neuronal_connectivity, seed, output):
    """ Generate N-G connectivity """
    # pylint: disable=redefined-argument-from-local,too-many-locals

    from voxcell import CellCollection

    from archngv.core.datasets import (
        NeuronalConnectivity,
        MicrodomainTesselation
    )
    from archngv.building.connectivity.neuroglial_generation import generate_neuroglial
    from archngv.building.exporters.edge_populations import neuroglial_connectivity

    from archngv.app.logger import LOGGER
    np.random.seed(seed)

    astrocytes_data = CellCollection.load(astrocytes)

    LOGGER.info('Generating neuroglial connectivity...')

    microdomains = MicrodomainTesselation(microdomains)

    data_iterator = generate_neuroglial(
        astrocytes=astrocytes_data,
        microdomains=microdomains,
        neuronal_connectivity=NeuronalConnectivity(neuronal_connectivity)
    )

    LOGGER.info('Exporting the per astrocyte files...')
    neuroglial_connectivity(
        data_iterator,
        neurons=CellCollection.load(neurons),
        astrocytes=astrocytes_data,
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

    from voxcell import CellCollection

    from archngv.building.exporters.edge_populations import bind_annotations as _run
    astrocytes = CellCollection.load(astrocytes)

    shutil.copyfile(input, output)
    _run(output, astrocytes=astrocytes, annotations_file=annotations)
