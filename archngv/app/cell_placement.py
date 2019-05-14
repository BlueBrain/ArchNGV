import click


@click.command()
@click.option("--config", help="Path to astrocyte placement YAML config", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--vasculature", help="Path to vasculature dataset", default=None)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def cmd(config, atlas, atlas_cache, vasculature, seed, output):
    """Generate astrocyte positions and radii"""
    import numpy as np

    from voxcell.nexus.voxelbrain import Atlas

    from archngv.core.vasculature_morphology import Vasculature
    from archngv.core.cell_placement.positions import create_positions
    from archngv.core.exporters import export_cell_placement_data

    from archngv.app.logger import LOGGER
    from archngv.app.utils import load_yaml

    config = load_yaml(config)

    atlas = Atlas.open(atlas, cache_dir=atlas_cache)
    voxelized_intensity = atlas.load_data(config['density'])
    voxelized_bnregions = atlas.load_data('brain_regions')

    assert np.issubdtype(voxelized_intensity.raw.dtype, np.floating)

    spatial_indexes = []
    if vasculature is not None:
        spatial_indexes.append(
            Vasculature.load(vasculature).spatial_index()
        )

    np.random.seed(seed)

    LOGGER.info('Generating cell positions / radii...')
    somata_positions, somata_radii = create_positions(
        config,
        voxelized_intensity,
        voxelized_bnregions,
        spatial_indexes=spatial_indexes
    )

    cell_names = ['GLIA_{:013d}'.format(index) for index in range(len(somata_positions))]
    cell_ids =  np.arange(len(cell_names))

    LOGGER.info('Export to CellData...')
    export_cell_placement_data(output, cell_ids, cell_names, somata_positions, somata_radii)

    LOGGER.info('Done!')