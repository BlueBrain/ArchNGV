"""
Generate microdomain tesselation
"""

import click


@click.command(help=__doc__)
@click.option("--config", help="Path to astrocyte microdomains YAML config", required=True)
@click.option("--cell-data", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output-dir", help="Path to output MVD3", required=True)
def cmd(config, cell_data, atlas, atlas_cache, seed, output_dir):
    # pylint: disable=missing-docstring,too-many-locals
    import os

    import numpy as np

    from scipy import stats
    from voxcell.nexus.voxelbrain import Atlas

    from archngv.core.data_structures.data_cells import CellData
    from archngv.core.exporters.export_microdomains import export_structure
    from archngv.core.microdomain import (
        generate_microdomain_tesselation,
        convert_to_overlappping_tesselation,
    )
    from archngv.spatial import BoundingBox

    from archngv.app.logger import LOGGER
    from archngv.app.utils import ensure_dir, load_yaml

    def _output_path(filename):
        return os.path.join(output_dir, filename)

    config = load_yaml(config)

    atlas = Atlas.open(atlas, cache_dir=atlas_cache)
    bbox = atlas.load_data('brain_regions').bbox
    bounding_box = BoundingBox(bbox[0], bbox[1])

    with CellData(cell_data) as data:
        somata_positions = data.astrocyte_positions[:]
        somata_radii = data.astrocyte_radii[:]

    ensure_dir(output_dir)

    np.random.seed(seed)

    LOGGER.info('Generating microdomains...')
    microdomains = generate_microdomain_tesselation(
        somata_positions, somata_radii, bounding_box
    )

    LOGGER.info('Export microdomains...')
    export_structure(_output_path('microdomains.h5'), microdomains)

    LOGGER.info('Generating overlapping microdomains...')
    overlap_distr = config['overlap_distribution']['values']
    overlap_distribution = stats.norm(loc=overlap_distr[0], scale=overlap_distr[1])
    overlapping_microdomains = convert_to_overlappping_tesselation(microdomains, overlap_distribution)

    LOGGER.info('Export overlapping microdomains...')
    export_structure(_output_path('overlapping_microdomains.h5'), overlapping_microdomains)

    LOGGER.info('Done!')
