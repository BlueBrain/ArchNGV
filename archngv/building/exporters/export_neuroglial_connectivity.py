""" Neuroglial connectivity expoerters functions """

import h5py
import libsonata
import numpy as np
import pandas as pd
import six

from archngv.core.connectivity_neuroglial import POPULATION_NAME


def _create_appendable_dataset(root, name, dtype, chunksize=1000):
    root.create_dataset(
        name, dtype=dtype, chunks=(chunksize,), shape=(0,), maxshape=(None,)
    )


def _append_to_dataset(dset, values):
    dset.resize(dset.shape[0] + len(values), axis=0)
    dset[-len(values):] = values


def export_neuroglial_connectivity(data_iterator, neurons, astrocytes, output_path):
    """
    Export the connectivity between neurons and astrocytes to SONATA Edges HDF5.

    Args:
        data_iterator: iterator yielding (astrocyte_id, synapses) pair,
            where `synapses` is pandas DataFrame with columns:
                - 'synapse_id' (as seen in the `synaptic_data`)
                - 'neuron_id' (postsynaptic neuron GID)
            `astrocyte_id`s appear in increasing order.
        neurons: voxcell.NodePopulation
        astrocytes: voxcell.NodePopulation

        output_path: path to output HDF5 file.
    """
    with h5py.File(output_path, 'w') as h5f:
        h5root = h5f.create_group('/edges/%s/' % POPULATION_NAME)
        h5group = h5root.create_group('0')

        # 'edge_type_id' is a required attribute storing index into CSV which we don't use
        _create_appendable_dataset(h5root, 'edge_type_id', dtype=np.int32)

        _create_appendable_dataset(h5root, 'source_node_id', dtype=np.uint64)
        _create_appendable_dataset(h5root, 'target_node_id', dtype=np.uint64)
        _create_appendable_dataset(h5group, 'synapse_id', dtype=np.uint64)

        prev_astrocyte_id = -1
        for astrocyte_id, synapses in data_iterator:
            assert astrocyte_id < astrocytes.size, 'astrocyte ID not within expected range'

            assert astrocyte_id > prev_astrocyte_id, 'astrocyte IDs do not appear in increasing order'
            prev_astrocyte_id = astrocyte_id

            if synapses.empty:
                continue

            assert synapses['neuron_id'].min() >= 0, 'neuron IDs not within expected range'
            assert synapses['neuron_id'].max() < neurons.size, 'neuron IDs not within expected range'

            synapses = synapses.sort_values(['neuron_id', 'synapse_id'])

            _append_to_dataset(
                h5root['edge_type_id'],
                np.full(len(synapses), -1, dtype=np.int32)
            )
            _append_to_dataset(
                h5root['target_node_id'],
                np.full(len(synapses), astrocyte_id, dtype=np.uint64)
            )
            _append_to_dataset(
                h5root['source_node_id'],
                synapses['neuron_id'].to_numpy()
            )
            _append_to_dataset(
                h5group['synapse_id'],
                synapses['synapse_id'].to_numpy()
            )

        h5root['source_node_id'].attrs['node_population'] = six.text_type(neurons.name)
        h5root['target_node_id'].attrs['node_population'] = six.text_type(astrocytes.name)

    # above, edge population has been sorted by (target_id, source_id)
    libsonata.EdgePopulation.write_indices(
        output_path,
        POPULATION_NAME,
        source_node_count=neurons.size,
        target_node_count=astrocytes.size
    )


def _load_annotations(h5_annotations, astrocytes, astrocyte_id):
    morph_name = astrocytes.attributes.loc[astrocyte_id, 'morphology']

    morph_group = h5_annotations[morph_name]

    synapse_group = morph_group['synapses']

    try:
        synapse_ids = synapse_group['ids'][:]
    except KeyError:
        return None

    assert len(np.unique(synapse_ids)) == len(synapse_ids), 'duplicate synapse IDs per astrocyte'

    return pd.DataFrame(
        {
            'section_id': synapse_group['morph_section_ids'][:],
            'segment_id': synapse_group['morph_segment_ids'][:],
            'segment_offset': synapse_group['morph_segment_offsets'][:],
        },
        index=synapse_ids
    )


def bind_annotations(h5_filepath, astrocytes, annotations_file):
    """ Bind synapse annotations with closest astrocyte sections. """

    # TODO: read / write with `libsonata` rather than direct HDF5 access
    with h5py.File(annotations_file, 'r') as h5_annotations, h5py.File(h5_filepath, 'a') as h5f:

        h5root = h5f['/edges/%s' % POPULATION_NAME]
        h5group = h5root['0']
        h5index = h5root['indices/target_to_source']

        edge_count = len(h5group['synapse_id'])

        h5group.create_dataset('morpho_section_id_post', shape=(edge_count,), dtype=np.int32)
        h5group.create_dataset('morpho_segment_id_post', shape=(edge_count,), dtype=np.int32)
        h5group.create_dataset('morpho_segment_offset_post', shape=(edge_count,), dtype=np.int32)

        for astrocyte_id, (r10, r11) in enumerate(h5index['node_id_to_ranges']):

            annotations = _load_annotations(h5_annotations, astrocytes, astrocyte_id)

            if annotations is None:
                continue

            assert r10 != r11, 'annotations exist for astrocyte with no synapses'

            assert r11 == r10 + 1, 'invalid edge range'
            r20, r21 = h5index['range_to_edge_id'][r10]
            assert r21 > r20, 'invalid edge range'

            synapse_ids = h5group['synapse_id'][r20:r21]

            assert np.all(sorted(synapse_ids) == sorted(annotations.index)), 'annotations mismatch synapse IDs'

            h5group['morpho_section_id_post'][r20:r21] = annotations.loc[synapse_ids, 'section_id']
            h5group['morpho_segment_id_post'][r20:r21] = annotations.loc[synapse_ids, 'segment_id']
            h5group['morpho_segment_offset_post'][r20:r21] = annotations.loc[synapse_ids, 'segment_offset']
