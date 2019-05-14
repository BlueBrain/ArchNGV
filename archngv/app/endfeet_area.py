import click


@click.command()
@click.option("--config", help="Path to YAML config", required=True)
@click.option("--vasculature-mesh", help="Path to vasculature mesh", required=True)
@click.option("--gliovascular-connectivity", help="Path to gliovascular connectivity (HDF5)", required=True)
@click.option("--gliovascular-data", help="Path to gliovascular data (HDF5)", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--parallel", help="Parallelize with 'multiprocessing'", is_flag=True, default=False)
@click.option("-o", "--output", help="Path to output file (HDF5)", required=True)
def cmd(config, vasculature_mesh, gliovascular_connectivity, gliovascular_data, seed, parallel, output):
    import numpy as np
    import openmesh

    from archngv.core.endfeet_area_reconstruction.area_generation import endfeet_area_generation
    from archngv.core.exporters.export_endfeet_areas import export_endfeet_areas

    from archngv.app.logger import LOGGER
    from archngv.app.utils import load_yaml

    config = load_yaml(config)

    LOGGER.info('Load vasculature mesh...')
    vasculature_mesh = openmesh.read_trimesh(vasculature_mesh)

    np.random.seed(seed)

    LOGGER.info('Setting up generator...')
    data_generator = endfeet_area_generation(
        mesh=vasculature_mesh,
        parameters=config,
        gliovascular_data_path=gliovascular_data,
        gliovascular_connectivity_path=gliovascular_connectivity,
        parallel=False
    )

    LOGGER.info("Export to HDF5...")
    export_endfeet_areas(output, data_generator)

    LOGGER.info('Done!')