"""
Generate neuroglial (N-G) connectivity
"""
import os
import click
import numpy as np


@click.group(help=__doc__)
def group():
    # pylint: disable=missing-docstring
    pass


@group.command()
@click.option("--neurons", help="Path to neuron node population (SONATA Nodes)", required=True)
@click.option("--astrocytes", help="Path to astrocyte node population (SONATA Nodes)", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--neuronal-connectivity", help="Path to neuron-neuron sonata edge file", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output file (SONATA Edges HDF5)", required=True)
def connectivity(neurons, astrocytes, microdomains, neuronal_connectivity, seed, output):
    """ Generate N-G connectivity """
    # pylint: disable=redefined-argument-from-local,too-many-locals

    from voxcell import CellCollection

    from archngv.core.datasets import (
        NeuronalConnectivity,
        MicrodomainTesselation
    )
    from archngv.building.connectivity.neuroglial_generation import generate_neuroglial
    from archngv.building.exporters.edge_populations import neuroglial_connectivity

    from archngv.app.logger import LOGGER
    np.random.seed(seed)

    astrocytes_data = CellCollection.load(astrocytes)

    LOGGER.info('Generating neuroglial connectivity...')

    microdomains = MicrodomainTesselation(microdomains)

    data_iterator = generate_neuroglial(
        astrocytes=astrocytes_data,
        microdomains=microdomains,
        neuronal_connectivity=NeuronalConnectivity(neuronal_connectivity)
    )

    LOGGER.info('Exporting the per astrocyte files...')
    neuroglial_connectivity(
        data_iterator,
        neurons=CellCollection.load(neurons),
        astrocytes=astrocytes_data,
        output_path=output
    )

    LOGGER.info("Done!")


def _properties_from_astrocyte(data):
    """Processes one astrocyte and returns annotation properties
    Args:
        data (dict): Dictionary with the following keys
            - index: astrocyte index
            - neurogial_connectivity: Path to ng connectivity sonata file
            - synaptic_data: Path to synaptic data sonata file
            - morphology_path: Path to morphology h5 file
            - morphology_position: Coordinates of morphology soma position

    Returns:
        tuple:
            connection_ids (np.ndarray): sonata edge ids
            section_ids (np.ndarray): Morphology section ids
            segment_ids: Morphology segment ids
            segment_offsets: Morphology segment offsets
    """
    from archngv.building.morphology_synthesis.annotation import annotate_synapse_location
    from archngv.core.datasets import NeuroglialConnectivity, NeuronalConnectivity
    from archngv.app.utils import readonly_morphology

    astrocyte_index = data['index']

    ng_connectivity = NeuroglialConnectivity(data['neuroglial_connectivity'])
    connection_ids = ng_connectivity.astrocyte_neuron_connections(astrocyte_index)

    if connection_ids.size == 0:
        return None

    synapse_ids = ng_connectivity.neuronal_synapses(connection_ids)

    synaptic_data = NeuronalConnectivity(data['synaptic_data'])
    synapse_positions = synaptic_data.synapse_positions(synapse_ids)

    morphology = readonly_morphology(data['morphology_path'], data['morphology_position'])
    locs = annotate_synapse_location(morphology, synapse_positions)

    return (
        connection_ids,
        locs.section_id.to_numpy(),
        locs.segment_id.to_numpy(),
        locs.segment_offset.to_numpy()
    )


class Worker:
    """Neuroglial properties helper"""
    def __init__(self, seed):
        self._seed = seed

    def __call__(self, data):

        seed = hash((self._seed, data['index'])) % (2 ** 32)
        np.random.seed(seed)

        return _properties_from_astrocyte(data)


def _dispatch_data(astrocytes, paths):

    for astro_id in range(len(astrocytes)):

        morphology_name = astrocytes.get_property('morphology', ids=astro_id)[0]
        morphology_path = os.path.join(paths['morph_dir'], morphology_name + '.h5')
        morphology_pos = astrocytes.positions(index=astro_id)[0]

        data = {
            'index': astro_id,
            'morphology_path': morphology_path,
            'morphology_position': morphology_pos
        }

        data.update(paths)

        yield data


def _neuroglial_properties(seed, astrocytes, n_connections, paths, map_func):
    """
    Args:
        seed (int): Random generator's seed
        astrocytes (CellData): node population of astrocytes
        n_connections (int): number of astrocyte-neuron connections
        paths (dict): dictionary with paths
        map_func (Callable): parallelization function

    Returns:
        properties (dict): Dictionary with string keys
            morph_section_id (np.ndarray): Array of ints corresponding to the
                astrocyte section id associated with each connected synapse
            morph_segment_id (np.ndarray): Array of ints corresponding to the
                astrocyte segment id associated with each connected synapse
            morph_segment_offset (np.ndarray): Array of floats corresponding
                to the segment offset associated with each connected synapse
    """

    properties = {
        'efferent_section_id': np.empty(n_connections, dtype=np.uint32),
        'efferent_segment_id': np.empty(n_connections, dtype=np.uint32),
        'efferent_segment_offset': np.empty(n_connections, dtype=np.float32)
    }

    it_results = filter(
        lambda result: result is not None,
        map_func(Worker(seed), _dispatch_data(astrocytes, paths))
    )

    for ids, section_ids, segment_ids, segment_offsets in it_results:

        properties['efferent_section_id'][ids] = section_ids
        properties['efferent_segment_id'][ids] = segment_ids
        properties['efferent_segment_offset'][ids] = segment_offsets

    return properties


@group.command()
@click.option("--input-file", help="Path to input file (SONATA Edges HDF5)", required=True)
@click.option("--output-file", help="Path to output file (SONATA Edges HDF5)", required=True)
@click.option("--astrocytes", help="Path to astrocyte node population (SONATA Nodes)", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--synaptic-data", help="Path to HDF5 with synapse positions", required=True)
@click.option("--morph-dir", help="Path to morphology folder", required=True)
@click.option("--parallel", help="Parallelize with 'multiprocessing'", is_flag=True, default=False)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
def finalize(input_file, output_file, astrocytes, microdomains, synaptic_data, morph_dir, parallel, seed):
    """For each astrocyte-neuron connection annotate the closest morphology section, segment, offset
    for each synapse.

    It adds in neuroglial connectivity the following properties:
        - efferent_section_id: int32
        - efferent_segment_id: int32
        - efferent_segment_offset: float32
    """
    import shutil
    from archngv.core.datasets import CellData, NeuroglialConnectivity
    from archngv.building.exporters.edge_populations import add_properties_to_edge_population
    from archngv.app.utils import apply_parallel_function

    paths = {
        'microdomains': microdomains,
        'synaptic_data': synaptic_data,
        'neuroglial_connectivity': input_file,
        'morph_dir': morph_dir
    }

    ng_connectivity = NeuroglialConnectivity(input_file)

    properties = _neuroglial_properties(
        seed=seed,
        astrocytes=CellData(astrocytes),
        n_connections=len(ng_connectivity),
        paths=paths,
        map_func=apply_parallel_function if parallel else map
    )

    shutil.copyfile(input_file, output_file)

    # add the new properties to the copied out file
    add_properties_to_edge_population(output_file, ng_connectivity.name, properties)
