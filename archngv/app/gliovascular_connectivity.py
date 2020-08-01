"""
Generate gliovascular (G-V) connectivity
"""

import click


@click.command(help=__doc__)
@click.option("--config", help="Path to astrocyte placement YAML config", required=True)
@click.option("--astrocytes", help="Path to the sonata file with astrocyte's positions", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--vasculature", help="Path to vasculature dataset", required=True)
@click.option("--vasculature-sonata", help="Path to vasculature sonata dataset", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--output", help="Path to output edges HDF5 (data)", required=True)
def cmd(config, astrocytes, microdomains, vasculature, vasculature_sonata, seed, output):
    # pylint: disable=missing-docstring,redefined-argument-from-local,too-many-locals
    import numpy as np
    from voxcell import CellCollection

    from archngv.core.datasets import (
        Vasculature,
        MicrodomainTesselation
    )

    from archngv.building.connectivity.gliovascular_generation import generate_gliovascular
    from archngv.building.exporters.edge_populations import gliovascular_connectivity as export

    from archngv.app.logger import LOGGER
    from archngv.app.utils import load_yaml

    LOGGER.info('Seed: %d', seed)
    np.random.seed(seed)

    params = load_yaml(config)
    vasculature = Vasculature.load(vasculature)

    LOGGER.info('Generating gliovascular connectivity...')

    astrocyte_positions = CellCollection.load_sonata(astrocytes).positions
    astrocyte_idx = np.arange(len(astrocyte_positions), dtype=np.int64)
    microdomains = MicrodomainTesselation(microdomains)

    (
        endfoot_surface_positions,
        endfeet_to_astrocyte_mapping,
        endfeet_to_vasculature_mapping
    ) = generate_gliovascular(astrocyte_idx, astrocyte_positions, microdomains, vasculature, params)

    LOGGER.info('Exporting sonata edges...')
    export(
        output,
        CellCollection.load_sonata(astrocytes),
        CellCollection.load_sonata(vasculature_sonata),
        endfeet_to_astrocyte_mapping,
        endfeet_to_vasculature_mapping,
        endfoot_surface_positions,
    )

    LOGGER.info("Done!")
