import os
import h5py
import logging
from . import helpers
from getpass import getuser
from voxcell import VoxelData

import numpy as np

from archngv.core.vasculature_morphology import Vasculature
from archngv.core.cell_placement.positions import create_positions
from archngv.core.util.bounding_box import BoundingBox
from archngv.core.exporters import export_cell_placement_data



L = logging.getLogger(__name__)


def get_spatial_index_list(ngv_config):

    from morphspatial import RTree
    from rtree.core import RTreeError

    static_spatial_indexes = []

    try:

        static_spatial_indexes.append(RTree.load(ngv_config.output_paths('vasculature_index')))

    except KeyError:
        L.info('Vasculature spatial index path not found. Proceeding without it.')
    except RTreeError:
        L.info('Vasculature Index Failed to be read. Proceeding without it')
    except AttributeError:
        L.info('Vasculature Index Failed to be read. Proceeding without it')

    if ngv_config._config['use_somata_geometry']:

        L.info('Neuronal somata geometry for indexing is enabled.')

        try:

            static_spatial_indexes.append(RTree.load(ngv_config.output_paths('neuronal_somata_index')))

        except KeyError:
            L.info('Neuronal somata index was not found. Proceeding without it.')
        except RTreeError:
            L.info('Neuronal Somata Index Failed to be read. Proceeding without it')
        except AttributeError:
            L.info('Neuronal Somata Index Failed to be read. Proceeding without it')

    return static_spatial_indexes


class OnSSDs(object):
    """ Decorator for automatically running an out of core index on the ssd disk drives
    of bb5. One has to allocate with --Cnvme to gain access to the /nvme directory.
    Spatial indexes on hdds are copied to the ssds and symbolic links are made at the index
    local directory that map to the nvme ones. The paths are modified in the config and then
    modified back when the processing is done. This way the project itself is agnostic on where
    the indexes really are located.
    """

    def __init__(self, config, index_keys):

        self._config = config
        self._keys = index_keys

    @property
    def ssd_directory(self):
        return os.path.join('/nvme', getuser(), os.environ["SLURM_JOB_ID"])

    @property
    def hdd_directory(self):
        return self._config.spatial_index_directory

    def config_entry(self, key):
        return self._config._config['output_paths'][key]

    def copy_with_ext(self, source, target, ext):
        cmd = 'rsync --ignore-existing {0}.{2} {1}.{2}'.format(source, target, ext)

        L.info('Executing command: ' + cmd)
        os.system('rsync --ignore-existing {0}.{2} {1}.{2}'.format(source, target, ext))

    def create_link(self, source, target, ext):

        try:
            # unlink any stray links
            os.unlink(target + '_ssd.' + ext)

        except OSError:
            pass

        L.info('Link {} -> {}'.format(source + '.dat', target + '_ssd.dat'))

        # create symbolic link in order for the files to be seen local to the project
        os.symlink(source + '.' + ext, target + '_ssd.' + ext)


    def __enter__(self):

        for key in self._keys:

            index_corpus = self.config_entry(key).replace('spatial_index/', '')

            L.info('Index Corpus: {}'.format(index_corpus))

            ssd_path = os.path.join(self.ssd_directory, index_corpus)
            hdd_path = os.path.join(self.hdd_directory, index_corpus)

            self.copy_with_ext(hdd_path, ssd_path, 'dat')
            self.copy_with_ext(hdd_path, ssd_path, 'idx')

            self.create_link(ssd_path, hdd_path, 'dat')
            self.create_link(ssd_path, hdd_path, 'idx')

            self._config._config['output_paths'][key] += '_ssd'

    def __exit__(self, *args):

        for key in self._keys:

            index_corpus = self.config_entry(key).replace('spatial_index/', '')

            hdd_link_path = os.path.join(self.hdd_directory, index_corpus)
            self._config._config['output_paths'][key] = self._config._config['output_paths'][key].replace('_ssd', '')

            os.unlink(hdd_link_path + '.dat')
            os.unlink(hdd_link_path + '.idx')

def create_cell_positions(ngv_config, run_parallel):
    """
    Given the vasculature, bounding_box and the config generate positions and radii
    of astrocytes.

    Args:
        ngv_config (NGVConfig):
            Contains the parameters, input and output absolute paths that
            are required for the circuit generation.

        vasculature (Vasculature):
            Vasculature Data structure.

        bounding_box (BoundingBox):
            The placement will take place in this bounding space

    Returns:
        positions (2D array)
        radiis (1D array)

    """


    cell_placement_parameters = ngv_config.parameters['cell_placement']

    voxelized_intensity = VoxelData.load_nrrd(ngv_config.input_paths('voxelized_intensity'))
    voxelized_bnregions = VoxelData.load_nrrd(ngv_config.input_paths('voxelized_brain_regions'))

    vasculature = Vasculature.load(ngv_config.input_paths('vasculature'))

    # placement bounding box
    bounding_box = BoundingBox.from_voxel_data(voxelized_bnregions)

    voxelized_intensity.raw = voxelized_intensity.raw.astype(np.float)

    static_spatial_indexes = get_spatial_index_list(ngv_config)

    # astro somata pos and radii
    somata_positions, somata_radii = create_positions(cell_placement_parameters,
                                                      voxelized_intensity,
                                                      voxelized_bnregions,
                                                      spatial_indexes=static_spatial_indexes)

    cell_names = ['GLIA_{:013d}'.format(index) for index in range(len(somata_positions))]
    cell_ids =  range(len(cell_names))

    export_cell_placement_data(ngv_config.output_paths('cell_data'), cell_ids, cell_names, somata_positions, somata_radii)

    astro_collection = helpers.create_astrocyte_collection(ngv_config, cell_names, somata_positions, somata_radii)

    L.info('Saving mvd3 circuit')

    # save the circuit to file
    astro_collection.save_mvd3(ngv_config.output_paths('circuit'))

"""
if __name__ == '__main__':
    import sys
    from archngv import NGVConfig

    config_path = sys.argv[1]

    config = NGVConfig.from_file(config_path)

    keys = ['vasculature_index']
    if config._config['use_somata_geometry']:
        keys.append('neuronal_somata_index')

    with OnSSDs(config, keys) as _:
        create_cell_positions(config, False)
"""
