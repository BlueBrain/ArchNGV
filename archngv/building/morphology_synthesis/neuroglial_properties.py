"""Astrocyte morphology properties that are attached to the neuroglial connectivity
following synthesis
"""
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple

import numpy as np

from archngv.app.utils import readonly_morphology
from archngv.building.morphology_synthesis.annotation import annotate_synapse_location
from archngv.core.datasets import NeuroglialConnectivity, NeuronalConnectivity

if TYPE_CHECKING:

    from pandas import DataFrame

    from archngv.core.datastructures import CellData


def astrocyte_morphology_properties(
    seed: int,
    astrocytes: "CellData",
    n_connections: int,
    paths: Dict[str, Path],
    map_function=map,
) -> Dict[str, np.ndarray]:
    """
    Args:
        seed: Random generator's seed
        astrocytes: node population of astrocytes
        n_connections: number of astrocyte-neuron connections
        paths: dictionary with paths
        map_func: parallelization function

    Returns:
        A dict mapping string keys to numpy arrays. The following keys are returned:
            - astrocyte_section_id: Array of ints corresponding to the
                astrocyte section id associated with each connected synapse
            - astrocyte_segment_id: Array of ints corresponding to the
                astrocyte segment id associated with each connected synapse
            - astrocyte_segment_offset: Array of floats corresponding
                to the segment offset associated with each connected synapse
    """
    properties = {
        "astrocyte_section_id": np.empty(n_connections, dtype=np.uint32),
        "astrocyte_segment_id": np.empty(n_connections, dtype=np.uint32),
        "astrocyte_segment_offset": np.empty(n_connections, dtype=np.float32),
        "astrocyte_section_pos": np.empty(n_connections, dtype=np.float32),
    }

    it_results = filter(
        lambda result: result is not None,
        map_function(NeuroglialWorker(seed), _dispatch_neuroglial_data(astrocytes, paths)),
    )

    for ids, df_locations in it_results:

        properties["astrocyte_section_id"][ids] = df_locations.section_id
        properties["astrocyte_segment_id"][ids] = df_locations.segment_id
        properties["astrocyte_segment_offset"][ids] = df_locations.segment_offset
        properties["astrocyte_section_pos"][ids] = df_locations.section_position

    return properties


class NeuroglialWorker:
    """Neuroglial properties helper"""

    def __init__(self, seed: int):
        self._seed = seed

    def __call__(self, data: Dict[str, Any]) -> Optional[Tuple[np.ndarray, "DataFrame"]]:

        seed = hash((self._seed, data["index"])) % (2**32)
        np.random.seed(seed)

        return _properties_from_astrocyte(data)


def _dispatch_neuroglial_data(
    astrocytes: "CellData", paths: Dict[str, Path]
) -> Iterator[Dict[str, Any]]:
    """Dispatches data to parallell workers"""
    for astro_id in range(len(astrocytes)):

        morphology_name = astrocytes.get_property("morphology", ids=astro_id)[0]
        morphology_path = str(Path(paths["morph_dir"], morphology_name + ".h5"))
        morphology_pos = astrocytes.positions(index=astro_id)[0]

        data = {
            "index": astro_id,
            "morphology_path": morphology_path,
            "morphology_position": morphology_pos,
        }

        data.update(paths)

        yield data


def _properties_from_astrocyte(data: dict) -> Optional[Tuple[np.ndarray, "DataFrame"]]:
    """Processes one astrocyte and returns annotation properties
    Args:
        data: Dictionary with the following keys
            - index: astrocyte index
            - neurogial_connectivity: Path to ng connectivity sonata file
            - synaptic_data: Path to synaptic data sonata file
            - morphology_path: Path to morphology h5 file
            - morphology_position: Coordinates of morphology soma position

    Returns:
        A tuple (connections_ids, locations_dataframe) where connection_ids are the
        neuron ids to which the astrocyte with `astrocyte_index` connects to, and
        locations_daraframe a dataframe with one row per connection id and the following
        columns:
            - section_id: Morphology's section id where the connection is established.
            - segment_id: Morphology segment id where the connection is established.
            - segment_offset: Fractional offset along the segment.

        If there are no connection_ids ofr current astrocyte, None will be returned.

        Note: There is one connection id per synapse, therefore the same astrocyte -
        neuron pair can be present multiple times. because an astrocyte may encapsulate
        multiple synapses of the same neuron.
    """
    astrocyte_index = data["index"]

    ng_connectivity = NeuroglialConnectivity(data["neuroglial_connectivity"])
    connection_ids = ng_connectivity.astrocyte_neuron_connections(astrocyte_index)

    if connection_ids.size == 0:
        return None

    synapse_ids = ng_connectivity.neuronal_synapses(connection_ids)

    synaptic_data = NeuronalConnectivity(data["synaptic_data"])
    synapse_positions = synaptic_data.synapse_positions(synapse_ids)

    morphology = readonly_morphology(data["morphology_path"], data["morphology_position"])
    locations_dataframe = annotate_synapse_location(morphology, synapse_positions)

    return connection_ids, locations_dataframe
