import click


@click.group()
def app():
    """Back and forth format conversion"""


@app.command()
@click.option("-i", "--input", help="Path to cell data (HDF5)", required=True)
@click.option("--output-format", type=click.Choice(['mvd3']), help="Output format", required=True)
@click.option("-o", "--output", help="Path to output file", required=True)
def cell_data(input, output_format, output):
    import numpy as np
    import pandas as pd

    from voxcell import CellCollection
    from archngv.core.data_structures.data_cells import CellData

    cells = CellCollection()
    with CellData(input) as data:
        np.testing.assert_equal(
            data.astrocyte_gids,
            np.arange(len(data))
        )
        cells.positions = data.astrocyte_positions[:]
        cells.properties = pd.DataFrame({
            'radius': data.astrocyte_radii[:],
            'morphology': data.astrocyte_names[:],
        })

    cells.seeds = [0, 0, 0, 0]
    cells.save_mvd3(output)
