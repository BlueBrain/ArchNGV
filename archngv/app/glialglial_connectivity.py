"""
Generate gliovascular (G-V) connectivity
"""

import click


@click.command(help=__doc__)
@click.option("--astrocytes", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--touches-dir", help="Path to touches directory", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--output-connectivity", help="Path to output HDF5 (connectivity)", required=True)
def cmd(astrocytes, touches_dir, seed, output_connectivity):
    # pylint: disable=missing-docstring,redefined-argument-from-local,too-many-locals
    import numpy as np
    from archngv.core.datasets import CellData
    from archngv.building.connectivity.glialglial import generate_glialglial
    from archngv.building.exporters.edge_populations import glialglial_connectivity

    from archngv.app.logger import LOGGER

    LOGGER.info('Seed: %d', seed)
    np.random.seed(seed)

    LOGGER.info('Creating symmetric connections from touches...')
    astrocyte_data = generate_glialglial(touches_dir)

    LOGGER.info('Exporting to SONATA file...')
    glialglial_connectivity(astrocyte_data, len(CellData(astrocytes)), output_connectivity)

    LOGGER.info("Done!")
