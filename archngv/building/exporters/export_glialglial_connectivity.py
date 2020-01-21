"""
Generate glialglial (G-G) connectivity
"""

import logging
import h5py
import libsonata
import numpy as np
import six

from archngv.core.connectivity_glialglial import POPULATION_NAME


L = logging.getLogger(__name__)


def export_glialglial_connectivity(astrocyte_data, n_astrocytes, output_path):
    """
    Export he connectivity between glia and glia to SONATA Edges HDF5

    Args:

        astrocyte_data: DataFrame, with the columns:
        - 'astrocyte_source_id'
        - 'astrocyte_target_id'
        - 'connection_id'
        sorted by ['astrocyte_source_id', 'astrocyte_target_id', 'connection_id']
        n_astrocytes: The number of astrocytes
        output_path: path to output HDF5 file
    """
    with h5py.File(output_path, 'w') as h5f:

        h5root = h5f.create_group(f'/edges/{POPULATION_NAME}')

        # 'edge_type_id' is a required attribute storing index into CSV which we don't use
        h5root.create_dataset('edge_type_id',
                              data=np.full(len(astrocyte_data), -1, dtype=np.int32))

        h5root.create_dataset('source_node_id',  # astrocytes
                              data=astrocyte_data['astrocyte_source_id'].to_numpy(), dtype=np.uint64)
        h5root.create_dataset('target_node_id',  # astrocytes
                              data=astrocyte_data['astrocyte_target_id'].to_numpy(), dtype=np.uint64)

        h5group = h5root.create_group('0')
        h5group.create_dataset('connection_id',
                               data=astrocyte_data['connection_id'], dtype=np.uint64)

        h5root['source_node_id'].attrs['node_population'] = six.text_type('astrocytes')
        h5root['target_node_id'].attrs['node_population'] = six.text_type('astrocytes')

    # above, edge population has been sorted by (target_id, source_id)
    libsonata.EdgePopulation.write_indices(
        output_path,
        POPULATION_NAME,
        source_node_count=n_astrocytes,
        target_node_count=n_astrocytes
    )
