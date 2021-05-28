""" Neuroglial connectivity exporter functions """
import logging
import h5py
import libsonata
import numpy as np
import pandas as pd

from archngv.exceptions import NGVError
from archngv.core.constants import Population

L = logging.getLogger(__name__)


def add_properties_to_edge_population(filepath, population_name, properties):
    """Add properties that are not already existing to an edge population.

    Args:
        filepath (str): EdgePopulation h5 file path
        populaton_name (str): The name of the EdgePopulation
        properties (dict): A dict with property names as keys and 1D numpy arrays
            as values.

    Raises:
        AssertionError: If property name exists or if property values length is not
            compatible with the edge population
    """
    with h5py.File(filepath, 'r+') as h5f:

        group = h5f[f'/edges/{population_name}/0']
        length = h5f[f'/edges/{population_name}/source_node_id'].shape[0]

        for name, values in properties.items():

            if name in group:
                raise NGVError(f"'{name}' property already exists.")

            if values.size != length:
                raise NGVError(f"Incompatible length. Expected: {length}. Given: {values.size}")

            group.create_dataset(name, data=values)
            L.info('Added edge Property: %s', name)


def _write_edge_population(output_path,
                           source_population_name, target_population_name,
                           source_population_size, target_population_size,
                           source_node_ids, target_node_ids,
                           edge_population_name,
                           edge_properties):
    # pylint: disable=too-many-arguments

    assert len(source_node_ids) == len(target_node_ids)

    with h5py.File(output_path, 'w') as h5f:
        h5root = h5f.create_group(f'/edges/{edge_population_name}')

        # 'edge_type_id' is a required attribute storing index into CSV which we don't use
        h5root.create_dataset(
            'edge_type_id', data=np.full(len(source_node_ids), -1, dtype=np.int32))

        h5root.create_dataset('source_node_id', data=source_node_ids, dtype=np.uint64)
        h5root.create_dataset('target_node_id', data=target_node_ids, dtype=np.uint64)

        h5group = h5root.create_group('0')

        # add edge properties
        for name, values in edge_properties.items():
            h5group.create_dataset(name, data=values)
            L.info('Added edge Property: %s', name)

        h5root['source_node_id'].attrs['node_population'] = str(source_population_name)
        h5root['target_node_id'].attrs['node_population'] = str(target_population_name)

    if len(source_node_ids) > 0:
        L.info('Creating edge indexing in: %s', edge_population_name)
        # above, edge population has been sorted by (target_id, source_id)
        libsonata.EdgePopulation.write_indices(
            output_path,
            edge_population_name,
            source_node_count=source_population_size,
            target_node_count=target_population_size
        )
    else:
        L.warning('Indexing will not be done. No edges in: %s', edge_population_name)


def neuroglial_connectivity(astrocyte_data, neurons, astrocytes, output_path):
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
    edge_properties = {'synapse_id': astrocyte_data['synapse_id'].to_numpy(dtype=np.uint64)}

    _write_edge_population(
        output_path=output_path,
        source_population_name=astrocytes.population_name,
        target_population_name=neurons.population_name,
        source_population_size=len(astrocytes.properties),
        target_population_size=len(neurons.properties),
        source_node_ids=astrocyte_data['astrocyte_id'],
        target_node_ids=astrocyte_data['neuron_id'],
        edge_population_name=Population.NEUROGLIAL,
        edge_properties=edge_properties
    )


def gliovascular_connectivity(output_path, astrocytes, vasculature, endfeet_to_astrocyte,
                              endfeet_to_vasculature, endfoot_surface_positions):
    """Creation of the gliovascular connectivity."""

    # require these datasets to be the same size
    assert len(endfeet_to_astrocyte) == len(endfeet_to_vasculature) == len(
        endfoot_surface_positions)

    # endfoot ids are the positional indices from these datasets
    endfoot_ids = np.arange(len(endfeet_to_astrocyte), dtype=np.uint64)
    astrocyte_ids = endfeet_to_astrocyte

    # get the section/segment ids and use them to get the vasculature node ids
    vasculature_properties = vasculature.properties.loc[:, ['section_id', 'segment_id']]
    vasculature_properties["index"] = vasculature_properties.index
    vasculature_properties = vasculature_properties.set_index(['section_id', 'segment_id'])

    indices = pd.MultiIndex.from_arrays(endfeet_to_vasculature.T)
    vasculature_ids = vasculature_properties.loc[indices, "index"].to_numpy()

    edge_properties = {
        'endfoot_id': endfoot_ids,
        'endfoot_surface_x': endfoot_surface_positions[:, 0].astype(np.float32),
        'endfoot_surface_y': endfoot_surface_positions[:, 1].astype(np.float32),
        'endfoot_surface_z': endfoot_surface_positions[:, 2].astype(np.float32),
        'vasculature_section_id': endfeet_to_vasculature[:, 0].astype(np.uint32),
        'vasculature_segment_id': endfeet_to_vasculature[:, 1].astype(np.uint32),
    }

    _write_edge_population(
        output_path=output_path,
        source_population_name=Population.VASCULATURE,
        target_population_name=Population.ASTROCYTES,
        source_population_size=len(vasculature.properties),
        target_population_size=len(astrocytes.properties),
        source_node_ids=vasculature_ids,
        target_node_ids=astrocyte_ids,
        edge_population_name=Population.GLIOVASCULAR,
        edge_properties=edge_properties
    )


def glialglial_connectivity(glialglial_data, n_astrocytes, output_path):
    """
    Export he connectivity between glia and glia to SONATA Edges HDF5

    Args:

        glialglial_data: DataFrame, with the touch properties
        sorted by ['astrocyte_source_id', 'astrocyte_target_id', 'connection_id']
        n_astrocytes: The number of astrocytes
        output_path: path to output HDF5 file
    """

    _write_edge_population(
        output_path=output_path,
        source_population_name=Population.ASTROCYTES,
        target_population_name=Population.ASTROCYTES,
        source_population_size=n_astrocytes,
        target_population_size=n_astrocytes,
        source_node_ids=glialglial_data.pop('source_node_id').to_numpy(),
        target_node_ids=glialglial_data.pop('target_node_id').to_numpy(),
        edge_population_name=Population.GLIALGLIAL,
        edge_properties=glialglial_data
    )
