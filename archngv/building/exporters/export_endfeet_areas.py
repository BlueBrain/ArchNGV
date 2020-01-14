""" Endfeetome exporters """
import logging

import h5py
import numpy as np

L = logging.getLogger(__name__)


def export_endfeet_areas(filepath, data_generator):
    """ Endfeetome """
    with h5py.File(filepath, 'w') as fd:

        metadata = fd.create_group('metadata')

        metadata.attrs['object_type'] = 'endfoot_mesh'

        meshes = fd.create_group('objects')

        for endfoot_index, points, triangles, thickness in data_generator:

            mesh_group = meshes.create_group('endfoot_{}'.format(endfoot_index))

            mesh_group.create_dataset('points', data=points)
            mesh_group.create_dataset('triangles', data=triangles)

            mesh_group.attrs.create('thickness', thickness, dtype=np.float32)

            L.info('written %d', endfoot_index)