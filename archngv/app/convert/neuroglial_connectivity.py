"""
Convert neuroglial connectivity to SONATA Edges
"""

import itertools
import os.path

from collections import defaultdict

import click
import h5py
import numpy as np
import pandas as pd


def _astrocyte_ids(size, offsets):
    result = np.full(size, -1, dtype=np.int64)
    for _id, (r0, r1) in enumerate(zip(offsets[:-1], offsets[1:])):
        result[r0:r1] = _id
    assert not np.any(result < 0)
    return result.astype(np.uint64)


def _extract_connectivity(h5f, connectome):
    from bluepy.sonata.connectome import POST_GID

    result = pd.DataFrame()
    synapse_ids = h5f['/Astrocyte/synapse'][:].astype(np.uint64)
    result['synapse_id'] = synapse_ids
    result['source_node_id'] = connectome.synapse_properties(synapse_ids, POST_GID).values.astype(np.uint64)
    result['target_node_id'] = _astrocyte_ids(len(result), h5f['/Astrocyte/offsets'][:, 0])
    return result


def _load_annotations(morph_dir, morph_names):
    def _load_chunk(astrocyte_id, morph_name):
        filepath = os.path.join(morph_dir, '%s_synapse_annotation.h5' % morph_name)
        with h5py.File(filepath, 'r') as h5f:
            loc = h5f['synapse_location']
            return pd.DataFrame({
                'astrocyte_id': astrocyte_id,
                'synapse_id': h5f['synapse_id'][:],
                'section_id': loc['section_id'][:],
                'segment_id': loc['segment_id'][:],
                'segment_offset': loc['segment_offset'][:],
            })

    return pd.concat([
        _load_chunk(_id, morph_name)
        for _id, morph_name in morph_names.iteritems()
    ])


def _bind_annotations(df, astrocytes, morph_dir):
    astrocyte_ids = df['target_node_id'].unique()
    morph_names = astrocytes.attributes.loc[astrocyte_ids, 'morphology']
    annotations = _load_annotations(morph_dir, morph_names)

    annotations.set_index(['astrocyte_id', 'synapse_id'], inplace=True)
    annotations.rename(columns={
        'section_id': 'morpho_section_id_post',
        'segment_id': 'morpho_segment_id_post',
        'segment_offset': 'morpho_offset_segment_post',
    }, inplace=True)

    result = df.join(annotations, on=('target_node_id', 'synapse_id'), how='inner')
    assert len(result) == len(df)

    return result


def _write_index(node_ids, max_id, out):
    offset = 0
    grouped = defaultdict(list)
    for value, group in itertools.groupby(node_ids):
        count = sum(1 for _ in group)
        grouped[value].append((offset, offset + count))
        offset += count
    index1, index2 = [], []
    for node_id in range(max_id):
        rng2 = grouped.get(node_id, [])
        rng1 = (len(index2), len(index2) + len(rng2))
        index1.append(rng1)
        index2.extend(rng2)
    out.create_dataset('node_id_to_ranges', data=index1, dtype=np.uint64)
    out.create_dataset('range_to_edge_id', data=index2, dtype=np.uint64)


def _write_edge_population(df, name, source, target, filepath):
    import six

    with h5py.File(filepath, 'w') as out:
        root = out.create_group('/edges/%s' % name)
        group = root.create_group('0')

        for prop, column in df.iteritems():
            if prop in ('source_node_id', 'target_node_id'):
                root[prop] = column
            else:
                group[prop] = column

        root['source_node_id'].attrs['node_population'] = six.text_type(source.name)
        root['target_node_id'].attrs['node_population'] = six.text_type(target.name)

        # 'edge_type_id' is a required attribute storing index into CSV which we don't use
        root['edge_type_id'] = np.full(len(df), -1, dtype=np.int32)

        _write_index(
            root['source_node_id'][:],
            max_id=source.size,
            out=root.create_group('indices/source_to_target')
        )
        _write_index(
            root['target_node_id'][:],
            max_id=target.size,
            out=root.create_group('indices/target_to_source')
        )


@click.command(help=__doc__)
@click.option("--astrocytes", help="Path to astrocyte node population (SONATA Nodes)", required=True)
@click.option("--morph-dir", help="Path to astrocyte morphologies folder", required=True)
@click.option("--connectivity", help="Path to neuroglial connectivity HDF5", required=True)
@click.option("--circuit", help="Path to base circuit config (SONATA)", required=True)
@click.option("-o", "--output", help="Path to output file (SONATA Edges)", required=True)
def cmd(astrocytes, morph_dir, connectivity, circuit, output):
    # pylint: disable=missing-docstring
    from bluepy.sonata import Circuit
    from voxcell.sonata import NodePopulation

    from archngv.app.logger import LOGGER
    from archngv.app.utils import choose_connectome

    astrocytes = NodePopulation.load(astrocytes)
    connectome = choose_connectome(Circuit(circuit))

    LOGGER.info("Converting connectivity information...")
    with h5py.File(connectivity, 'r') as h5f:
        df = _extract_connectivity(h5f, connectome)

    LOGGER.info("Binding synapse annotations...")
    df = _bind_annotations(df, astrocytes, morph_dir)

    LOGGER.info("Export to SONATA Edges...")
    # TODO: sort with an external tool
    df.sort_values(['target_node_id', 'source_node_id', 'synapse_id'], inplace=True)
    _write_edge_population(
        df,
        name='neuroglial',
        source=connectome.target,
        target=astrocytes,
        filepath=output
    )

    LOGGER.info("Done!")
