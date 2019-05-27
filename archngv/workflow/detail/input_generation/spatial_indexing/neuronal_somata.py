import os
import logging
import itertools

import h5py
import numpy

from .spatial_index_adapter import spatial_index

from morphspatial import BoundingBox
from voxcell import VoxelData

import helpers

L = logging.getLogger(__name__)


def _test_spatial_index(ngv_config, bounding_box):

    L.info("Test Spatial index is being generated")
    tsi_filename = os.path.join(ngv_config.experiment_directory, 'si_test')

    L.info("Test si wiil be stored at {}".format(tsi_filename))

    ps = bounding_box.center + numpy.random.random((2, 3))
    rs = numpy.random.uniform(1, 2, size=2)

    si = spatial_index(tsi_filename)
    return si.create_from_spheres(ps, rs)


def _neuronal_somata_as_spheres(nrn_points, nrn_radii, nsi_filename):
    L.info('Neuronal point data is going to be used for indexing.')
    si = spatial_index(nsi_filename)
    return si.create_from_spheres(nrn_points, nrn_radii)


def _neuronal_somata_as_convex_polygons(ngv_config, filepath, nrn_idx, nrn_points):

    L.info('Geometry of neuronal somata will be used for indexing.')
    with h5py.File(ngv_config.input_paths("somata_geometry"), 'r') as fp:

        idx    = set(nrn_idx)
        n_keys = len(fp.keys())

        shape_points = [fp[str(n)][:] + nrn_points[n] for n in xrange(n_keys) if n in idx]

        si = spatial_index(filepath)
        return si.create_from_convex_polygons(shape_points)


def _subsample_neurons(ngv_config, n_neurons):

    fraction = float(ngv_config._config['neuronal_subsample'])

    L.info('Keeping {}% of neurons'.format(fraction * 100.))

    idx = numpy.arange(n_neurons, dtype=numpy.int)
    cidx = numpy.random.choice(idx, int(fraction * idx.size), replace=False)

    return numpy.in1d(idx, cidx)


def _output_neuronal_gids(ngv_config, nrn_idx):

    neuronal_gids = nrn_idx  + 1

    gid_out = os.path.join(ngv_config.morphology_directory, 'neuronal_gids.np')
    numpy.save(gid_out, neuronal_gids)

    L.info('{} gids were written to file.'.format(neuronal_gids.size))


def neuronal_somata_spatial_index(ngv_config, bounding_box):

    L.info("Neuronal Somata Spatial Index is being generated.")

    nsi_filename = ngv_config.output_paths('neuronal_somata_index')

    L.info("nsi will be stored at {}".format(nsi_filename))

    nrn_point_data = numpy.load(ngv_config.input_paths('microcircuit_point_data'))

    if 'neuronal_subsample' in ngv_config._config:

        subsample_mask = _subsample_neurons(ngv_config, len(nrn_point_data))

    else:

        subsample_mask = numpy.ones(len(nrn_point_data), dtype=numpy.bool)

    assert nrn_point_data.shape[1] == 4

    nrn_points = nrn_point_data[:, :3]
    nrn_radii  = nrn_point_data[:, 3]

    # inside bounding box of vasculature
    mask_inside = bounding_box.spheres_inside(nrn_points, nrn_radii)

    L.info('{} neuronal somata from total {} in bbox'.format(mask_inside.sum(), mask_inside.size))
    mask = mask_inside & subsample_mask

    L.info('Neuronal somata after subsampling: {}'.format(mask.sum()))
    nrn_idx = numpy.where(mask)[0]

    if ngv_config._config['use_somata_geometry']:

        neuronal_spatial_index = _neuronal_somata_as_convex_polygons(ngv_config, nsi_filename, nrn_idx, nrn_points)

    else:

        neuronal_spatial_index = _neuronal_somata_as_spheres(nrn_points[mask], nrn_radii[mask], nsi_filename)

    _output_neuronal_gids(ngv_config, nrn_idx)

    return neuronal_spatial_index


def create_neuronal_somata_spatial_index(ngv_config, map_func):

    voxelized_bnregions = VoxelData.load_nrrd(ngv_config.input_paths('voxelized_brain_regions'))

    bounding_box = BoundingBox.from_voxel_data(voxelized_bnregions)

    return neuronal_somata_spatial_index(ngv_config, bounding_box)


if __name__ == '__main__':
    import sys
    from archngv import NGVConfig

    config_filepath = sys.argv[1]
    create_neuronal_somata_spatial_index(NGVConfig.from_file(config_filepath), map)

