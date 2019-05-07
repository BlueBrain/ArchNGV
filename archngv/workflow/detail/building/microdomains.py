import h5py
import logging
from scipy import stats

from archngv.core.microdomain import generate_microdomain_tesselation
from archngv.core.microdomain import convert_to_overlappping_tesselation
from archngv.core.exporters.export_microdomains import export_mesh
from archngv.core.exporters.export_microdomains import export_structure
from archngv.core.data_structures.data_cells import CellData
from voxcell import VoxelData
from archngv.core.util.bounding_box import BoundingBox


from . import helpers


L = logging.getLogger(__name__)


def create_microdomains(ngv_config, run_parallel):
    """ Generate microdomain tesselation convert to an overlapping one.
    """

    parameters = ngv_config.parameters['microdomain_tesselation']

    voxelized_bnregions = VoxelData.load_nrrd(ngv_config.input_paths('voxelized_brain_regions'))

    # placement bounding box
    bounding_box = BoundingBox.from_voxel_data(voxelized_bnregions)

    with CellData(ngv_config.output_paths('cell_data')) as data:

        somata_positions = data.astrocyte_positions
        somata_radii = data.astrocyte_radii

        microdomain_tesselation = generate_microdomain_tesselation(somata_positions, somata_radii, bounding_box)

    L.info('Generating Microdomains')
 
    export_structure(ngv_config.output_paths('microdomain_structure'),
                     microdomain_tesselation,
                     global_coordinate_system=False)

    export_structure(ngv_config.output_paths('microdomain_structure').replace('.h5', '_global.h5'),
                     microdomain_tesselation,
                     global_coordinate_system=True)

    export_mesh(microdomain_tesselation, ngv_config.output_paths('microdomain_mesh'))

    overlap_distr = parameters['overlap_distribution']['values']
    overlap_distribution = stats.norm(loc=overlap_distr[0], scale=overlap_distr[1])

    L.info('Generating overlapping microdomains')

    overlapping_tesselation = convert_to_overlappping_tesselation(microdomain_tesselation, overlap_distribution)
    export_structure(ngv_config.output_paths('overlapping_microdomain_structure'),
                     overlapping_tesselation,
                     global_coordinate_system=False)

    export_structure(ngv_config.output_paths('overlapping_microdomain_structure').replace('.h5', '_global.h5'),
                     overlapping_tesselation,
                     global_coordinate_system=True)

    export_mesh(overlapping_tesselation, ngv_config.output_paths('overlapping_microdomain_mesh'))
