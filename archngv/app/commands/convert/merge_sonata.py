"""Merge SONATA Nodes/Edges

This is temporary solution needed as per:
https://bbpteam.epfl.ch/project/issues/browse/NGV-132?focusedCommentId=99967&page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel#comment-99967
"""
import os

from voxcell import CellCollection
import click
import h5py
import libsonata
import numpy as np
import pandas as pd

from archngv.app.logger import LOGGER as L
from archngv.app.utils import ensure_dir, REQUIRED_PATH


# this population name is used to show that this a weird concatenation of
# what should be different sonata populations, since they have different
# attributes
POPULATION_NAME = 'AllMerged'
DEFAULT_VALUE = -1


def _create_appendable_dataset(root, name, dtype, chunksize=1024):
    return root.create_dataset(
        name, dtype=dtype, chunks=(chunksize,), shape=(0,), maxshape=(None,)
    )


def _append_to_dataset(dset, values):
    dset.resize(dset.shape[0] + len(values), axis=0)
    dset[-len(values):] = values


def _open_default_population(storage):
    populations = storage.population_names
    assert len(populations) == 1, 'Need to only have a single population'
    population = storage.open_population(next(iter(populations)))
    return population


def _fill_dataset(dataset, size):
    _append_to_dataset(dataset, np.full(size, DEFAULT_VALUE).astype(dataset.dtype))


def _merge_edges(edges_paths, output_path):
    L.info('Merging edges...')
    seen_attributes, offsets = {}, {}
    with h5py.File(output_path, 'w') as h5:
        root = h5.create_group('/edges/%s/' % POPULATION_NAME)
        group = root.create_group('0')
        total_population = 0

        source_ids = _create_appendable_dataset(root, 'source_node_id', dtype=np.uint64)
        target_ids = _create_appendable_dataset(root, 'target_node_id', dtype=np.uint64)

        for edges in edges_paths:
            offsets[edges] = total_population
            population = _open_default_population(libsonata.EdgeStorage(edges))
            selection = libsonata.Selection([(0, population.size), ])

            _append_to_dataset(source_ids, population.source_nodes(selection))
            _append_to_dataset(target_ids, population.target_nodes(selection))

            for attr in population.attribute_names:
                values = population.get_attribute(attr, selection)

                if attr not in seen_attributes:
                    seen_attributes[attr] = _create_appendable_dataset(
                        group, attr, dtype=values.dtype)
                    if total_population:
                        _fill_dataset(seen_attributes[attr], total_population)

                _append_to_dataset(seen_attributes[attr], values)

            for missing_attr in set(seen_attributes) - population.attribute_names:
                L.info('Filling up missing attribute: %s', missing_attr)
                _fill_dataset(seen_attributes[missing_attr], population.size)

            total_population += population.size

        # 'edge_type_id' is a required attribute storing index into CSV which we don't use
        root.create_dataset('edge_type_id', data=np.full(total_population, -1, dtype=np.int32))

        source_node_count = len(source_ids)
        target_node_count = len(target_ids)

    L.info('Writing indices...')
    libsonata.EdgePopulation.write_indices(
        output_path,
        POPULATION_NAME,
        source_node_count,
        target_node_count
    )

    return offsets


def _merge_nodes(nodes_paths, output_path):
    L.info('Merging nodes...')
    seen_attributes, offsets = {}, {}

    total_population = 0
    populations = []
    for nodes in nodes_paths:
        offsets[nodes] = total_population
        population = CellCollection.load_sonata(nodes).as_dataframe()
        if population.index[0] == 1:
            population.index = population.index - 1
        total_population += len(population)

        for attr in population:
            seen_attributes[attr] = population[attr].dtype

        populations.append(population)

    populations = pd.concat(populations, ignore_index=True, sort=False)

    new_population = CellCollection(POPULATION_NAME)
    new_population.positions = populations[['x', 'y', 'z']].to_numpy()
    new_population.properties = pd.DataFrame(index=range(total_population))

    for attr in seen_attributes:
        if attr in 'xyz':  # already handled in positions
            continue

        values = np.nan_to_num(populations[attr].to_numpy(dtype=seen_attributes[attr]),
                               DEFAULT_VALUE)
        new_population.properties[attr] = values

    new_population.save(output_path)

    return offsets


def _fix_edges(nodes_offsets, edges_offsets, output):
    L.info('Fixing edges...')
    with h5py.File(output, 'r+') as h5:
        root = h5['/edges/%s/' % POPULATION_NAME]
        # astrocytes have moved in the nodes list to be stacked on the end of
        # the neuron nodes
        root['source_node_id'][edges_offsets:] += nodes_offsets


@click.command(help=__doc__)
@click.option("--n2n-nodes", required=True, type=REQUIRED_PATH,
              help="Path to neuron-to-neuron sonata node file")
@click.option("--n2n-edges", required=True, type=REQUIRED_PATH,
              help="Path to neuron-to-neuron sonata edge file")
@click.option("--a2n-nodes", required=True, type=REQUIRED_PATH,
              help="Path to astrocyte-to-neuron sonata node file")
@click.option("--a2n-edges", required=True, type=REQUIRED_PATH,
              help="Path to astrocyte-to-neuron sonata edge file")
@click.option("-o", "--output", required=True,
              help="Path to output directory")
def cmd(n2n_nodes, n2n_edges, a2n_nodes, a2n_edges, output):
    '''merge neuron and astrocyte sonata files'''
    ensure_dir(output)

    nodes_offsets = _merge_nodes([n2n_nodes, a2n_nodes, ], os.path.join(output, 'nodes.sonata'))
    edges_offsets = _merge_edges([n2n_edges, a2n_edges, ], os.path.join(output, 'edges.sonata'))

    _fix_edges(nodes_offsets[a2n_nodes], edges_offsets[a2n_edges],
               os.path.join(output, 'edges.sonata'))
