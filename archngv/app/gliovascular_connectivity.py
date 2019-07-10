"""
Generate gliovascular (G-V) connectivity
"""

import click


@click.command(help=__doc__)
@click.option("--config", help="Path to astrocyte placement YAML config", required=True)
@click.option("--cell-data", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--vasculature", help="Path to vasculature dataset", default=None)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--output-data", help="Path to output HDF5 (data)", required=True)
@click.option("--output-connectivity", help="Path to output HDF5 (connectivity)", required=True)
def cmd(config, cell_data, microdomains, vasculature, seed, output_data, output_connectivity):
    # pylint: disable=missing-docstring,redefined-argument-from-local,too-many-locals
    import numpy as np

    from archngv.core.connectivity.gliovascular_generation import generate_gliovascular
    from archngv.core.data_structures.data_cells import CellData
    from archngv.core.data_structures.data_microdomains import MicrodomainTesselation
    from archngv.core.exporters import export_gliovascular_data, export_gliovascular_connectivity
    from archngv.core.data_structures.vasculature_morphology import Vasculature

    from archngv.app.logger import LOGGER
    from archngv.app.utils import load_yaml

    params = load_yaml(config)
    vasculature = Vasculature.load(vasculature)

    np.random.seed(seed)

    LOGGER.info('Generating gliovascular connectivity...')

    with CellData(cell_data) as cell_data:
        somata_positions = cell_data.astrocyte_positions[:]

    n_astrocytes = len(somata_positions)
    somata_idx = np.arange(len(somata_positions), dtype=np.uintp)

    with MicrodomainTesselation(microdomains) as microdomains:
        (
            endfeet_surface_positions,
            endfeet_graph_positions,
            endfeet_to_astrocyte_mapping,
            endfeet_to_vasculature_mapping
        ) = generate_gliovascular(somata_idx, somata_positions, microdomains, vasculature, params)

        LOGGER.info('Exporting gliovascular data...')
        export_gliovascular_data(
            output_data,
            endfeet_surface_positions,
            endfeet_graph_positions
        )

        LOGGER.info('Exporting gliovascular connectivity...')
        export_gliovascular_connectivity(
            output_connectivity,
            n_astrocytes,
            endfeet_to_astrocyte_mapping,
            endfeet_to_vasculature_mapping
        )

    LOGGER.info("Done!")
