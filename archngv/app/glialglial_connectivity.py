"""
Generate gliovascular (G-V) connectivity
"""

import click


@click.command(help=__doc__)
@click.option("--cell-data", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--touches-dir", help="Path to touches directory", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--output-connectivity", help="Path to output HDF5 (connectivity)", required=True)
def cmd(cell_data, touches_dir, seed, output_connectivity):
    # pylint: disable=missing-docstring,redefined-argument-from-local,too-many-locals
    import numpy as np
    from archngv import CellData
    from archngv.building.connectivity.glialglial import generate_glialglial
    from archngv.building.exporters import export_glialglial_connectivity

    from archngv.app.logger import LOGGER

    np.random.seed(seed)

    LOGGER.info('Creating symmetric connections from touches...')

    astrocyte_data = generate_glialglial(touches_dir)

    LOGGER.info('Exporting to SONATA file...')

    n_astrocytes = len(CellData(cell_data))
    export_glialglial_connectivity(astrocyte_data, n_astrocytes, output_connectivity)

    LOGGER.info("Done!")
