"""
Generate endfeet area geometry
"""

import click


@click.command(help=__doc__)
@click.option("--config", help="Path to YAML config", required=True)
@click.option("--vasculature-mesh", help="Path to vasculature mesh", required=True)
@click.option("--gliovascular-connectivity", help="Path to sonata gliovascular file", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output file (HDF5)", required=True)
def cmd(config, vasculature_mesh, gliovascular_connectivity, seed, output):
    # pylint: disable=missing-docstring
    import numpy as np
    import openmesh

    from archngv.core.datasets import GliovascularConnectivity
    from archngv.building.endfeet_reconstruction.area_generation import endfeet_area_generation
    from archngv.building.exporters.export_endfeet_areas import export_endfeet_areas

    from archngv.app.logger import LOGGER
    from archngv.app.utils import load_yaml

    np.random.seed(seed)
    LOGGER.info('Seed: %d', seed)

    config = load_yaml(config)

    LOGGER.info('Load vasculature mesh at %s', vasculature_mesh)
    vasculature_mesh = openmesh.read_trimesh(vasculature_mesh)

    gliovascular_connectivity = GliovascularConnectivity(gliovascular_connectivity)
    endfeet_points = gliovascular_connectivity.vasculature_surface_targets

    LOGGER.info('Setting up generator...')
    data_generator = endfeet_area_generation(
        vasculature_mesh=vasculature_mesh,
        parameters=config,
        endfeet_points=endfeet_points
    )

    LOGGER.info("Export to HDF5...")
    export_endfeet_areas(output, data_generator, len(endfeet_points))

    LOGGER.info('Done!')
