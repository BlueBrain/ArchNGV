"""
Convert CellData to SONATA Nodes
"""

import click


@click.command(help=__doc__)
@click.option("-i", "--input", help="Path to cell data (HDF5)", required=True)
@click.option("-o", "--output", help="Path to output file (SONATA Nodes HDF5)", required=True)
def cmd(input, output):
    # pylint: disable=missing-docstring,redefined-builtin
    import numpy as np

    from voxcell import CellCollection
    from archngv.core.datasets import CellData

    with CellData(input) as data:
        n_cells = len(data)
        np.testing.assert_equal(
            data.astrocyte_gids,
            np.arange(n_cells)
        )
        cells = CellCollection(population_name='astrocytes')

        # TODO: to reset to 'data.astrocyte_positions[:]' once the synthesis pb are solved
        cells.positions = np.zeros((n_cells, 3), dtype=np.float32)

        cells.properties['radius'] = data.astrocyte_radii
        cells.properties['morphology'] = data.astrocyte_names

        cells.properties['mtype'] = "ASTROCYTE"
        cells.properties["model_type"] = "biophysical"

    cells.save_sonata(output)
