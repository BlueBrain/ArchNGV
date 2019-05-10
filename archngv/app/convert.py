import click


@click.group()
def app():
    """Back and forth format conversion"""


@app.command()
@click.option("-i", "--input", help="Path to cell data (HDF5)", required=True)
@click.option("--population", help="Population name", required=True)
@click.option("-o", "--output", help="Path to output file (SONATA Nodes HDF5)", required=True)
def cell_data(input, population, output):
    import numpy as np
    import pandas as pd

    from voxcell.sonata import NodePopulation
    from archngv.core.data_structures.data_cells import CellData

    with CellData(input) as data:
        n_cells = len(data)
        np.testing.assert_equal(
            data.astrocyte_gids,
            np.arange(n_cells)
        )
        result = NodePopulation(population, size=n_cells)
        result.positions = data.astrocyte_positions[:]
        result.attributes['radius'] = data.astrocyte_radii[:]
        result.attributes['morphology'] = data.astrocyte_names[:]

    result.save(output)
