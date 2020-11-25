"""High level Circuit container. All NGV data can be accessed from here.

It uses bluepysnap.circuit.Circuit as main backend.
"""
from pathlib import Path
from cached_property import cached_property

import numpy as np

from bluepysnap import Circuit
from bluepysnap.nodes import NodeStorage, NodePopulation
from bluepysnap.edges import EdgeStorage, EdgePopulation
from bluepysnap.morph import MorphHelper

from archngv.core.structures import Atlas, Microdomains
from archngv.core.constants import Population
from archngv.exceptions import NGVError


def _find_config_location(path):
    """Returns the path to the config location.

    If a circuit directory is given instead, it looks for the config at
    circuit_dir/build/ngv_config.json.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise NGVError(f'Config path does not exist: {path}')

    return path if path.is_file() else path / 'build/ngv_config.json'


def _dispatch_nodes(population_type):
    """Dispatch function returning the correct node class using a population type as input.

    Args:
        population_type(str): one of the population type defined in the ngv config file in the cells
            section.
    Returns:
        NGVNodes: a subclass of NGVNodes constructor.
    """
    mapping = {"protoplasmic_astrocytes": Astrocytes,
               Population.NEURONS: Neurons,
               Population.VASCULATURE: Vasculature,
               }
    return mapping.get(population_type, NGVNodes)


def _dispatch_edges(population_type):
    """Dispatch function returning the correct edge class using a population type as input.

    Args:
        population_type(str): one of the population type defined in the ngv config file in the
            connectivities section.
    Returns:
        NGVEdges: a subclass of NGVEdges constructor.
    """
    mapping = {Population.NEURONAL: Neuronal,
               Population.GLIOVASCULAR: GlioVascular,
               Population.NEUROGLIAL: NeuroGlial,
               Population.GLIALGLIAL: GlialGlial
               }
    return mapping.get(population_type, NGVEdges)


def _dispatch(population_type, storage):
    """Dispatch function returning the correct node/edge class using a population type as input.

    Args:
        storage (bluepysnap.nodes.NodeStorage/bluepysnap.edges.EdgeStorage): the storage containing
            the population to dispatch.
        population_type(str): one of the population type defined in the ngv config file in the
            cells or connectivities section.
    Returns:
        NGVNodes/NGVEdges: a subclass of NGVNodes/NGVEdges constructor.
    """
    if isinstance(storage, NodeStorage):
        return _dispatch_nodes(population_type)
    return _dispatch_edges(population_type)


def _collect_ngv_populations(partial_config, extra_config, cls):
    """The overloaded loading function for the different node/edges populations.

    This function is crucial to have a correct circuit behavior.

    Notes:
        This function overloads the standard bluepysnap.Circuit population loading and returns the
        extended ngv classes instead. This provides a compact way for inheritance of the standard
        bluepysnap Node/EdgePopulation.
    """
    inv = {v["population"]: k for k, v in extra_config.items()}
    result = {}
    for file_config in partial_config:
        storage = cls(file_config)
        for population in storage.population_names:
            if population in result:
                raise NGVError("Duplicated population: '%s'" % population)
            pop_type = inv[population]
            result[population] = _dispatch(pop_type, storage)(storage, population,
                                                              extra_config[pop_type])
    return result


class AstrocytesMorphHelper(MorphHelper):
    """Specific class for accessing the astrocyte's morphologies.

    The astrocyte's morphologies cannot be stored as a swc file due to missing fields such as
    the missing astrocytes radius. This class aims at allowing the use of h5 morphology files for
    the astrocytes and returning the correct morphology path.
    """
    def get_filepath(self, node_id):
        """Return path to h5 morphology file corresponding to `node_id`."""
        return super().get_filepath(node_id).replace(".swc", ".h5", 1)


class NGVNodes(NodePopulation):
    """The standard NGV node population.

    This class aims at facilitate the functions renaming/addition while keeping the snap backend
    intact.
    """

    def __init__(self, node_storage, population_name, extra_conf=None):
        """Initializes a NGVNodes object from a NodeStorage, a population name and extra config.

        Args:
            node_storage (NodeStorage): the node storage containing the node population
            population_name (str): the name of the node population
            extra_conf (dict/None): ngv extra config from the cells section.

        Returns:
            NGVNodes: A NGVNodes object.
        """
        super().__init__(node_storage, population_name)
        self._extra_conf = extra_conf

    @cached_property
    def morphology(self):
        """Redirection of the standard bluepysnap.NodePopulation.morph."""
        return self.morph


class Astrocytes(NGVNodes):
    """Astrocyte nodes population.

    Notes:
        This class adds the extra objects needed for an astrocytes to the NGVNodes. The extras
        are the microdomains, specific to the astrocytes, that connect the astrocytes to the
        vasculature. The morphologies are produced in a different morphology directory
        (neuronal circuits can be read only) so the morphology is overloaded.
    """

    @cached_property
    def microdomains(self):
        """Access to the microdomain object for this astrocyte population."""
        return Microdomains(self._extra_conf["microdomains_file"],
                            self._extra_conf["microdomains_overlapping_file"])

    @cached_property
    def morph(self):
        """Access to the astrocyte morphologies.

        Notes:
            The morphologies are produced in a different morphology directory
            (neuronal circuits can be read only) so the morphology dir path is overloaded.
            Moreover the astrocytes morphologies are stored as h5 file and not swc (see :
            AstrocytesMorphHelper class).
        """
        return AstrocytesMorphHelper(self._extra_conf["morphologies_dir"], self)


class Vasculature(NGVNodes):
    """Vasculature node population.

    Notes:
        This class adds the extra objects needed for the vasculature to the NGVNodes. The extras
        are the initial vasculature moprhological file one can use through the
        Vasculature.morphology API. The vasculature's meshes are also available from the
        Vasculature.surface_mesh API.
    """

    @cached_property
    def morph(self):
        """Access to the Vasculature morphology.

        Returns:
            archngv.core.datasets.Vasculature: A wrapper of the more low level VasculatureApi.

        Notes:
            The morphologies of the vasculature is handled by the VasculatureAPI package.
        """
        from archngv.core.datasets import Vasculature as VasculatureMorphology
        return VasculatureMorphology.load(self._extra_conf["vasculature_file"])

    @cached_property
    def surface_mesh(self):
        """Returns vasculature surface mesh object."""
        import trimesh
        return trimesh.load(self._extra_conf["vasculature_mesh_file"])


class Neurons(NGVNodes):
    """Neurons node population."""


class NGVEdges(EdgePopulation):
    """The standard NGV edge population.

    This class aims at facilitate the functions renaming/addition while keeping the snap backend
    intact.
    """

    def __init__(self, edge_storage, population_name, extra_conf):
        """Initializes a NGVEdges object from a EdgeStorage, a population name and extra config.

        Args:
            edge_storage (EdgeStorage): the edge storage containing the edge population
            population_name (str): the name of the edge population
            extra_conf (dict/None): ngv extra config from the 'connectivities' section.

        Returns:
            NGVEdges: A NGVEdges object.
        """
        super().__init__(edge_storage, population_name)
        self._extra_conf = extra_conf


class GlioVascular(NGVEdges):
    """Gliovalcular edges API."""

    @property
    def astrocytes(self):
        """Returns the corresponding astrocyte population.

        Returns:
            Astrocytes: A Astrocytes NGVNodes subclass.
        """
        return self.target

    @property
    def vasculature(self):
        """Returns the corresponding vasculature population.

        Returns:
            Vasculature: A Vasculature NGVNodes subclass.
        """
        return self.source

    @cached_property
    def surface_meshes(self):
        """Access the endfeet surface meshes for the gliovascular connection."""
        from archngv.core.datasets import EndfootSurfaceMeshes
        return EndfootSurfaceMeshes(self._extra_conf['endfeet_areas'])

    def astrocyte_endfeet(self, astrocyte):
        """Returns the endfeet ids connected to an astrocyte."""
        return self.afferent_edges(astrocyte)

    def vasculature_endfeet(self, vasculature):
        """Returns the endfeet ids connected to a vasculature segment id."""
        return self.efferent_edges(vasculature)

    def connected_astrocytes(self, vasculature_group=None):
        """Returns all the astrocytes ids connected to the vasculature."""
        if vasculature_group is None:
            return np.unique(self._population.target_nodes(self._population.select_all()))
        return np.unique([target for _, target in self.iter_connections(vasculature_group, None)])

    def connected_vasculature(self, astrocyte_group=None):
        """Returns all the segment ids (vasculature node ids) connected to an astrocyte."""
        if astrocyte_group is None:
            return np.unique(self._population.source_nodes(self._population.select_all()))
        return np.unique([source for source, _ in self.iter_connections(None, astrocyte_group)])

    def vasculature_surface_targets(self, endfoot_ids):
        """Returns the Endfeet surface targets on vasculature."""

        return self.properties(endfoot_ids, properties=['endfoot_surface_x',
                                                        'endfoot_surface_y',
                                                        'endfoot_surface_z'])

    def vasculature_sections_segments(self, endfoot_ids):
        """Returns the Endfeet section segment from the vasculature."""
        return self.properties(endfoot_ids, properties=['efferent_section_id',
                                                        'efferent_segment_id'])


class Neuronal(NGVEdges):
    """Neuronal edges API."""


class GlialGlial(NGVEdges):
    """GlialGlial edges API."""

    def astrocyte_gap_junctions(self, astrocyte_id):
        """Returns the gap junctions (edges) connected to node:`astrocyte_id`."""
        res = np.append(self.efferent_edges(astrocyte_id), self.afferent_edges(astrocyte_id))
        # inplace sort for better perfs (much better for small arrays and bit better for large)
        # sort allows for chunked data queries in libsonata
        res.sort()
        return res

    def astrocyte_astrocytes(self, astrocyte_id):
        """Returns the astrocyte ids connected to the node:`astrocyte_id`.

        Args:
            astrocyte_id (int): the source astrocyte id.

        Returns:
            numpy.array: all the astrocyte ids connected to the astrocyte_id. The ids are for the
            ongoing or outgoing edges.

        Notes:
            uses the unique inside the e/afferent_nodes but from rapid bench it is quite
            similar to : np.unique(np.append(a, b)) or np.unique(np.append(np.unique(a),
            np.unique(b))). The last one being less memory consuming I kept this one.
            TODO : Need to test on a real circuit with touches.
        """
        return np.unique(np.append(self.efferent_nodes(astrocyte_id, unique=True),
                                   self.afferent_nodes(astrocyte_id, unique=True)))


class NeuroGlial(NGVEdges):
    """NeuroGlial edges API."""

    @property
    def neurons(self):
        """Returns the corresponding neuron population.

        Returns:
            Neurons: A Neurons NGVNodes subclass.
        """
        return self.target

    @property
    def astrocytes(self):
        """Returns the corresponding astrocyte population.

        Returns:
            Astrocytes: A Astrocytes NGVNodes subclass.
        """
        return self.source

    @property
    def synapses(self):
        """Returns the neuronal connectome to access the synapse population."""
        return self._edge_storage.circuit.neuronal_connectome

    def astrocyte_synapses(self, astrocyte_id):
        """Returns the synapse ids connected to a given `astrocyte_id`."""
        edge_ids = self.efferent_edges(astrocyte_id)
        return self.properties(edge_ids=edge_ids, properties="synapse_id").to_numpy()

    def astrocyte_synapses_properties(self, astrocyte_id, properties=None):
        """Returns the synapse properties and ids connected to a given `astrocyte_id`."""
        synapse_ids = self.astrocyte_synapses(astrocyte_id)
        return self._edge_storage.circuit.neuronal_connectome.properties(synapse_ids,
                                                                         properties=properties)

    def connected_neurons(self, astrocyte_group=None):
        """Returns all the neuron ids connected to the astrocytes."""
        if astrocyte_group is None:
            return np.unique(self._population.target_nodes(self._population.select_all()))
        return np.unique([target for _, target in self.iter_connections(astrocyte_group, None)])

    def connected_astrocytes(self, neuron_group=None):
        """Returns all the astrocytes ids connected to the neurons."""
        if neuron_group is None:
            return np.unique(self._population.source_nodes(self._population.select_all()))
        return np.unique([source for source, _ in self.iter_connections(None, neuron_group)])


class NGVCircuit(Circuit):
    """High level access to the NGV circuit."""

    def __init__(self, path):
        """Initializes a NGV circuit object from an extended SONATA config file.

        Args:
            path (str): Path to a extended SONATA config file. Can be a directory or absolute
                path to the circuit config file.

        Returns:
            NGVCircuit: A NGVCircuit object.
        """
        self.path = _find_config_location(path)
        super().__init__(self.path)

    @cached_property
    def nodes(self):
        """Access to node population(s)."""
        return _collect_ngv_populations(self._config['networks']['nodes'],
                                        self._config['cells'],
                                        lambda cfg: NodeStorage(cfg, self)
                                        )

    @cached_property
    def edges(self):
        """Access to edges population(s)."""
        return _collect_ngv_populations(self._config['networks']['edges'],
                                        self._config['connectivities'],
                                        lambda cfg: EdgeStorage(cfg, self)
                                        )

    def _get_population(self, config_key, object_name):
        """Returns the population using the extra info from the ngv config."""
        container = {"cells": self.nodes, "connectivities": self.edges}
        try:
            return container[config_key][self._config[config_key][object_name]["population"]]
        except KeyError as e:
            raise NGVError(
                "Cannot import the '{}' object. Please verify the '{}' part from the "
                "config and if you have added the correct files "
                "in the network.".format(object_name, config_key)) from e

    @cached_property
    def neurons(self):
        """Access to the neuron node population."""
        return self._get_population("cells", Population.NEURONS)

    @cached_property
    def astrocytes(self):
        """Access to the astrocyte node population."""
        return self._get_population("cells", "protoplasmic_astrocytes")

    @cached_property
    def vasculature(self):
        """Access to the vasculature node population."""
        return self._get_population("cells", Population.VASCULATURE)

    @cached_property
    def neuronal_connectome(self):
        """Access to the neuronal connectivity edge population."""
        return self._get_population("connectivities", Population.NEURONAL)

    @cached_property
    def gliovascular_connectome(self):
        """Access to the gliovascular connectivity edge population."""
        return self._get_population("connectivities", Population.GLIOVASCULAR)

    @cached_property
    def neuroglial_connectome(self):
        """Access to the neuroglial connectivity edge population."""
        return self._get_population("connectivities", Population.NEUROGLIAL)

    @cached_property
    def glialglial_connectome(self):
        """Access to the glialglial connectivity edge population."""
        return self._get_population("connectivities", Population.GLIALGLIAL)

    @cached_property
    def atlases(self):
        """Access to the Voxel atlases"""
        config = self._config["atlases"]
        return {name: Atlas(name, filepath) for name, filepath in config.items()}

    def __repr__(self):
        """Representation of circuit."""
        return f"<NGVCircuit : {self.path}>"
