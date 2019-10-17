""" Neuroglial connectivity expoerters functions """
import logging
import h5py
import libsonata
import numpy as np
import pandas as pd
import six

from archngv.core.connectivity_neuroglial import POPULATION_NAME

L = logging.getLogger(__name__)


def export_neuroglial_connectivity(astrocyte_data, neurons, astrocytes, output_path):
    """
    Export the connectivity between neurons and astrocytes to SONATA Edges HDF5.

    Args:
        astrocyte_data: DataFrame, with the columns:
        - 'synapse_id' (as seen in the `synaptic_data`)
        - 'neuron_id' (postsynaptic neuron GID)
        - 'astrocyte_id': ie: pre-side
        sorted by ['neuron_id', 'astrocyte_id', 'synapse_id']
        neurons: voxcell.NodePopulation
        astrocytes: voxcell.NodePopulation

        output_path: path to output HDF5 file.
    """
    with h5py.File(output_path, 'w') as h5f:
        h5root = h5f.create_group('/edges/%s/' % POPULATION_NAME)

        # 'edge_type_id' is a required attribute storing index into CSV which we don't use
        h5root.create_dataset('edge_type_id',
                              data=np.full(len(astrocyte_data), -1, dtype=np.int32))

        h5root.create_dataset('source_node_id',  # astrocyte
                              data=astrocyte_data['astrocyte_id'].to_numpy(), dtype=np.uint64)
        h5root.create_dataset('target_node_id',  # post-synaptic neuron
                              data=astrocyte_data['neuron_id'].to_numpy(), dtype=np.uint64)

        h5group = h5root.create_group('0')
        h5group.create_dataset('synapse_id',
                               data=astrocyte_data['synapse_id'].to_numpy(), dtype=np.uint64)

        h5root['source_node_id'].attrs['node_population'] = six.text_type(astrocytes.name)
        h5root['target_node_id'].attrs['node_population'] = six.text_type(neurons.name)

    # above, edge population has been sorted by (target_id, source_id)
    libsonata.EdgePopulation.write_indices(
        output_path,
        POPULATION_NAME,
        source_node_count=astrocytes.size,
        target_node_count=neurons.size
    )


def _load_annotations(annotations_path):
    ret = []
    with h5py.File(annotations_path, 'r') as h5:
        for glia_name in h5:
            synapses = h5[glia_name]['synapses']
            if(not len(synapses) or
               ('ids' in synapses and not len(synapses['ids']))
               ):
                continue

            synapse_ids = synapses['ids'][:]
            assert len(np.unique(synapse_ids)) == len(synapse_ids), \
                'duplicate synapse IDs per astrocyte'

            ret.append(pd.DataFrame({'morphology': glia_name,
                                     'synapse_id': synapse_ids,
                                     'section_id': synapses['morph_section_ids'][:],
                                     'segment_id': synapses['morph_segment_ids'][:],
                                     'segment_offset': synapses['morph_segment_offsets'][:],
                                     },
                                    ))

    return pd.concat(ret, ignore_index=True, sort=False)


def bind_annotations(output_path, astrocytes, annotations_file):
    """ Bind synapse annotations with closest astrocyte sections. """
    sections = (_load_annotations(annotations_file)
                .set_index(['morphology', 'synapse_id'])
                )
    # TODO: read / write with `libsonata` rather than direct HDF5 access
    with h5py.File(output_path, 'a') as h5:
        h5group = h5['/edges/%s/0' % POPULATION_NAME]

        synapses = (pd.DataFrame({'astro_id': h5['/edges/neuroglial/source_node_id'][:],
                                  'synapse_id': h5['/edges/neuroglial/0/synapse_id'][:],
                                  })
                    .join(astrocytes.to_dataframe(), on='astro_id')
                    .join(sections, on=['morphology', 'synapse_id'])
                    )

        L.warning('Writing section datasets')
        h5group.create_dataset('efferent_section_id',
                               data=synapses.section_id.to_numpy(), dtype=np.int32)
        h5group.create_dataset('efferent_segment_id',
                               data=synapses.segment_id.to_numpy(), dtype=np.int32)
        h5group.create_dataset('efferent_segment_offset',
                               data=synapses.segment_offset.to_numpy(), dtype=np.float32)
