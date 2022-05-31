"""High level Circuit container. All NGV data can be accessed from here.

It uses bluepysnap.circuit.Circuit as main backend.
"""
import warnings
from pathlib import Path

import numpy as np
import trimesh
import vascpy
from bluepysnap import Circuit
from bluepysnap.edges import EdgePopulation, EdgeStorage
from bluepysnap.morph import MorphHelper
from bluepysnap.nodes import NodePopulation, NodeStorage
from bluepysnap.sonata_constants import Edge
from cached_property import cached_property

from archngv.core.constants import Population
from archngv.core.datasets import Microdomains
from archngv.core.structures import Atlas
from archngv.exceptions import NGVError


def _find_config_location(path):
    """Returns the path to the config location.

    If a circuit directory is given instead, it looks for the config at
    circuit_dir/build/ngv_config.json.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise NGVError(f"Config path does not exist: {path}")

    return path if path.is_file() else path / "build/ngv_config.json"


def _dispatch_nodes(population_type):
    """Dispatch function returning the correct node class using a population type as input.

    Args:
        population_type(str): one of the population type defined in the ngv config file in the cells
            section.
    Returns:
        NGVNodes: a subclass of NGVNodes constructor.
    """
    mapping = {
        "protoplasmic_astrocytes": Astrocytes,  # for backward compatibility
        Population.ASTROCYTES: Astrocytes,
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
    mapping = {
        "neuronal": Neuronal,  # for backward compatibility
        "gliovascular": GlioVascular,  # for backward compatibility
        "neuroglial": NeuroGlial,  # for backward compatibility
        Population.NEURONAL: Neuronal,
        Population.GLIOVASCULAR: GlioVascular,
        Population.NEUROGLIAL: NeuroGlial,
        Population.GLIALGLIAL: GlialGlial,
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


def _collect_ngv_populations(partial_config, cls):
    """The overloaded loading function for the different node/edges populations.

    This function is crucial to have a correct circuit behavior.

    Notes:
        This function overloads the standard bluepysnap.Circuit population loading and returns the
        extended ngv classes instead. This provides a compact way for inheritance of the standard
        bluepysnap Node/EdgePopulation.
    """
    # ``item`` is either nodes or edges single config item
    result = {}
    for file_config in partial_config:
        storage = cls(file_config)
        for population in storage.population_names:
            if population in result:
                raise NGVError("Duplicated population: '%s'" % population)
            pop_type = file_config["populations"][population].get("type", "biophysical")
            result[population] = _dispatch(pop_type, storage)(
                storage, population, file_config["populations"][population]
            )
    return result


class AstrocytesMorphHelper(MorphHelper):
    """Specific class for accessing the astrocyte's morphologies.

    The astrocyte's morphologies cannot be stored as a swc file due to missing fields such as
    the missing astrocytes radius. This class aims at allowing the use of h5 morphology files for
    the astrocytes and returning the correct morphology path.
    """

    def get_filepath(self, node_id, extension="h5"):
        """Return path to h5 morphology file corresponding to `node_id`."""
        return str(super().get_filepath(node_id, extension=extension))


class NGVNodes(NodePopulation):
    """The standard NGV node population.

    This class aims at facilitate the functions renaming/addition while keeping the snap backend
    intact.
    """

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

        return Microdomains(self.config["microdomains_file"])

    @cached_property
    def morph(self):
        """Access to the astrocyte morphologies.

        Notes:
            The morphologies are produced in a different morphology directory
            (neuronal circuits can be read only) so the morphology dir path is overloaded.
            Moreover the astrocytes morphologies are stored as h5 file and not swc (see :
            AstrocytesMorphHelper class).
        """
        return AstrocytesMorphHelper(
            self.config.get("morphologies_dir"),
            self,
            self.config.get("alternate_morphologies"),
        )


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
            vascpy.SectionVasculature: A wrapper of the more low level VasculatureApi.

        Notes:
            The morphologies of the vasculature is handled by the VasculatureAPI package.
        """

        return vascpy.SectionVasculature.load(self.config["vasculature_file"])

    @cached_property
    def point_graph(self):
        """Access to the Vasculature point-edges representation

        Returns:
            vascpy.PointVasculature
        """

        return vascpy.PointVasculature.load_sonata(self._node_storage.h5_filepath)

    @cached_property
    def surface_mesh(self):
        """Returns vasculature surface mesh object."""
        return trimesh.load(self.config["vasculature_mesh_file"])


class Neurons(NGVNodes):
    """Neurons node population."""


class NGVEdges(EdgePopulation):
    """The standard NGV edge population.

    This class aims at facilitate the functions renaming/addition while keeping the snap backend
    intact.
    """


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

        if "endfeet_meshes" in self.config:
            warnings.warn(
                "Deprecated key 'endfeet_meshes' instead of 'endfeet_meshes_file' was encountered.",
                DeprecationWarning,
            )
            return EndfootSurfaceMeshes(self.config["endfeet_meshes"])

        return EndfootSurfaceMeshes(self.config["endfeet_meshes_file"])

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

        return self.get(
            endfoot_ids,
            properties=["endfoot_surface_x", "endfoot_surface_y", "endfoot_surface_z"],
        )

    def vasculature_sections_segments(self, endfoot_ids):
        """Returns:
        edge_id: The sonata Vasculature node id that corresponds to the
            edge id in core.Vasculature and VasculatureAPI.PointVasculature
        vasculature_section_id: The section id of the vasculature morphology
        vasculature_segment_id: The segment id of the vaculature morphology
        """
        return self.get(
            endfoot_ids,
            properties=[
                Edge.SOURCE_NODE_ID,
                "vasculature_section_id",
                "vasculature_segment_id",
            ],
        )


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
        return np.unique(
            np.append(
                self.efferent_nodes(astrocyte_id, unique=True),
                self.afferent_nodes(astrocyte_id, unique=True),
            )
        )


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
        return self.get(edge_ids=edge_ids, properties="synapse_id").to_numpy()

    def astrocyte_synapses_properties(self, astrocyte_id, properties=None):
        """Returns the synapse properties and ids connected to a given `astrocyte_id`."""
        synapse_ids = self.astrocyte_synapses(astrocyte_id)
        return self._edge_storage.circuit.neuronal_connectome.get(
            synapse_ids, properties=properties
        )

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
        return _collect_ngv_populations(
            self._config["networks"]["nodes"], lambda cfg: NodeStorage(cfg, self)
        )

    @cached_property
    def edges(self):
        """Access to edges population(s)."""
        return _collect_ngv_populations(
            self._config["networks"]["edges"], lambda cfg: EdgeStorage(cfg, self)
        )

    def _get_population(self, node_class):
        for pop in list(self.nodes.values()) + list(self.edges.values()):
            if isinstance(pop, node_class):
                return pop
        raise NGVError(
            f"Cannot import the '{node_class}' object. Please verify the config and if"
            " you have added the correct files in the network."
        )

    @cached_property
    def neurons(self):
        """Access to the neuron node population."""
        return self._get_population(Neurons)

    @cached_property
    def astrocytes(self):
        """Access to the astrocyte node population."""
        return self._get_population(Astrocytes)

    @cached_property
    def vasculature(self):
        """Access to the vasculature node population."""
        return self._get_population(Vasculature)

    @cached_property
    def neuronal_connectome(self):
        """Access to the neuronal connectivity edge population."""
        return self._get_population(Neuronal)

    @cached_property
    def gliovascular_connectome(self):
        """Access to the gliovascular connectivity edge population."""
        return self._get_population(GlioVascular)

    @cached_property
    def neuroglial_connectome(self):
        """Access to the neuroglial connectivity edge population."""
        return self._get_population(NeuroGlial)

    @cached_property
    def glialglial_connectome(self):
        """Access to the glialglial connectivity edge population."""
        return self._get_population(GlialGlial)

    @cached_property
    def atlases(self):
        """Access to the Voxel atlases"""
        config = self._config["atlases"]
        return {name: Atlas(name, filepath) for name, filepath in config.items()}

    def __repr__(self):
        """Representation of circuit."""
        return f"<NGVCircuit : {self.path}>"
