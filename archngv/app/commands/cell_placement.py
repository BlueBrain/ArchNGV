"""
Generate astrocyte positions and radii
"""

import click


@click.command(help=__doc__)
@click.option("--config", help="Path to astrocyte placement YAML config", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--vasculature", help="Path to vasculature dataset", default=None)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def cmd(config, atlas, atlas_cache, vasculature, seed, output):
    # pylint: disable=missing-docstring,too-many-locals
    import numpy as np

    from voxcell.nexus.voxelbrain import Atlas
    from vasculatureapi import SectionVasculature

    from archngv.building.cell_placement.positions import create_positions
    from archngv.building.exporters.node_populations import export_astrocyte_population
    from archngv.building.checks import assert_bbox_alignment

    from archngv.spatial import BoundingBox
    from archngv.app.logger import LOGGER
    from archngv.app.utils import load_yaml

    np.random.seed(seed)
    LOGGER.info("Seed: %d", seed)

    config = load_yaml(config)

    atlas = Atlas.open(atlas, cache_dir=atlas_cache)
    voxelized_intensity = atlas.load_data(config['density'])
    voxelized_bnregions = atlas.load_data('brain_regions')

    assert np.issubdtype(voxelized_intensity.raw.dtype, np.floating)

    spatial_indexes = []
    if vasculature is not None:

        from ngv_spatial_index import sphere_rtree

        vasc = SectionVasculature.load(vasculature).as_point_graph()

        assert_bbox_alignment(
            BoundingBox.from_points(vasc.points),
            BoundingBox(voxelized_intensity.bbox[0],
                        voxelized_intensity.bbox[1])
        )

        spatial_indexes.append(sphere_rtree(vasc.points, 0.5 * vasc.diameters))

    LOGGER.info('Generating cell positions / radii...')
    somata_positions, somata_radii = create_positions(
        config,
        voxelized_intensity,
        voxelized_bnregions,
        spatial_indexes=spatial_indexes
    )

    cell_names = ['GLIA_{:013d}'.format(index) for index in range(len(somata_positions))]

    LOGGER.info('Export to CellData...')
    export_astrocyte_population(output, cell_names, somata_positions, somata_radii, mtype="ASTROCYTE")

    LOGGER.info('Done!')
