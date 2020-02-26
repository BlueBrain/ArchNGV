"""
Generate endfeet area geometry
"""

import click


@click.command(help=__doc__)
@click.option("--config", help="Path to YAML config", required=True)
@click.option("--vasculature-mesh", help="Path to vasculature mesh", required=True)
@click.option("--gliovascular-data", help="Path to gliovascular data (HDF5)", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--parallel", help="Parallelize with 'multiprocessing'", is_flag=True, default=False)
@click.option("-o", "--output", help="Path to output file (HDF5)", required=True)
def cmd(config, vasculature_mesh, gliovascular_data, seed, parallel, output):
    # pylint: disable=missing-docstring
    import numpy as np
    import openmesh

    from archngv import GliovascularData
    from archngv.building.endfeet_reconstruction.area_generation import endfeet_area_generation
    from archngv.building.exporters.export_endfeet_areas import export_endfeet_areas

    from archngv.app.logger import LOGGER
    from archngv.app.utils import load_yaml

    np.random.seed(seed)
    config = load_yaml(config)

    LOGGER.info('Parallel: %s', parallel)

    LOGGER.info('Load vasculature mesh...')
    vasculature_mesh = openmesh.read_trimesh(vasculature_mesh)

    with GliovascularData(gliovascular_data) as gdata:
        endfeet_points = gdata.endfoot_surface_coordinates[:]

    LOGGER.info('Setting up generator...')
    data_generator = endfeet_area_generation(
        vasculature_mesh=vasculature_mesh,
        parameters=config,
        endfeet_points=endfeet_points
    )

    LOGGER.info("Export to HDF5...")
    export_endfeet_areas(output, data_generator, len(endfeet_points))

    LOGGER.info('Done!')
