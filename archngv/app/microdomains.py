import click


@click.command()
@click.option("--config", help="Path to astrocyte microdomains YAML config", required=True)
@click.option("--cell-data", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output-dir", help="Path to output MVD3", required=True)
def cmd(config, cell_data, atlas, atlas_cache, seed, output_dir):
    """Generate microdomain tesselations"""
    import os

    import numpy as np

    from scipy import stats
    from voxcell.nexus.voxelbrain import Atlas

    from archngv.core.data_structures.data_cells import CellData
    from archngv.core.exporters.export_microdomains import export_mesh, export_structure
    from archngv.core.microdomain import (
        generate_microdomain_tesselation,
        convert_to_overlappping_tesselation,
    )
    from archngv.core.util.bounding_box import BoundingBox

    from archngv.app.logger import LOGGER
    from archngv.app.utils import ensure_dir, load_yaml

    def _output_path(filename):
        return os.path.join(output_dir, filename)

    config = load_yaml(config)

    atlas = Atlas.open(atlas, cache_dir=atlas_cache)
    bounding_box = BoundingBox.from_voxel_data(atlas.load_data('brain_regions'))

    with CellData(cell_data) as data:
        somata_positions = data.astrocyte_positions[:]
        somata_radii = data.astrocyte_radii[:]

    ensure_dir(output_dir)

    np.random.seed(seed)

    LOGGER.info('Generating microdomains...')
    microdomain_tesselation = generate_microdomain_tesselation(somata_positions, somata_radii, bounding_box)
    microdomain_tesselation = generate_microdomain_tesselation(
        somata_positions, somata_radii, bounding_box
    )

    LOGGER.info('Export structure / mesh...')
    export_structure(
        _output_path('structure.h5'),
        microdomain_tesselation,
        global_coordinate_system=False
    )
    export_structure(
        _output_path('structure_global.h5'),
        microdomain_tesselation,
        global_coordinate_system=True
    )
    export_mesh(
        microdomain_tesselation,
        _output_path('tesselation.stl')
    )

    LOGGER.info('Generating overlapping microdomains...')
    overlap_distr = config['overlap_distribution']['values']
    overlap_distribution = stats.norm(loc=overlap_distr[0], scale=overlap_distr[1])
    overlapping_tesselation = convert_to_overlappping_tesselation(microdomain_tesselation, overlap_distribution)

    LOGGER.info('Export structure / mesh...')
    export_structure(
        _output_path('overlapping_structure.h5'),
        overlapping_tesselation,
        global_coordinate_system=False
    )
    export_structure(
        _output_path('overlapping_structure_global.h5'),
        overlapping_tesselation,
        global_coordinate_system=True
    )
    export_mesh(
        overlapping_tesselation,
        _output_path('overlapping_tesselation.stl')
    )

    LOGGER.info('Done!')