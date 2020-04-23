""" High level Circuit container. All NGV data can be accessed from here. """

from cached_property import cached_property

from bluepysnap import Circuit
from archngv.core.structures import Vasculature, Endfeetome, Microdomains, Atlas
from archngv.core.constants import Population
from archngv.exceptions import NGVError


def _add_astrocytes_information(node_population, config):
    """Adds the astrocytes informations

    Args:
        node_population(bluepysnap.NodePopulation): the node population to transform
        config(dict): the astrocyte information block

    Notes:
        This extends a node population into an "astrocyte population". This is probably temporary
        and will be handle by snap but the endfeetome and microdomains are not finished yet.
    """
    from bluepysnap.morph import MorphHelper

    setattr(node_population, "microdomains",
            Microdomains(config["microdomains_file"],
                             config["microdomains_overlapping_file"]))

    setattr(node_population, "endfeetome",
            Endfeetome(config["endfeet_file"], config["endfeet_data_file"]))

    # overload of the morph helper from snap to allow multiple morphology paths
    # the morph.get is broken due to the h5 path instead of swc
    setattr(node_population, "morph", MorphHelper(config['morphologies_dir'], node_population))


def _load_atlases(config):
    """Dynamically load atlases."""
    return {name: Atlas(name, filepath) for name, filepath in config.items()}


class NGVSnapCircuit:
    """NGV circuit object."""

    def __init__(self, sonata_config):
        """Initializes a circuit object from a SONATA config or SONATA extended config file.

        Args:
            sonata_config (str): Path to a SONATA or SONATA extended config file.

        Returns:
            NGVSnapCircuit: A NGVCircuit object.
        """
        self._circuit = Circuit(sonata_config)
        self._config = self._circuit.config["networks"]

    @cached_property
    def nodes(self):
        """Access to cell population(s)."""
        if Population.ASTROCYTES in self._config:
            for glial_config in self._config[Population.ASTROCYTES]:
                population = glial_config["population"]
                astro = self._circuit.nodes[population]
                _add_astrocytes_information(astro, glial_config)
        return self._circuit.nodes

    @cached_property
    def edges(self):
        """Access to connectome class(es)."""
        if "gliovascular" in self._config:
            from archngv.core.connectivities import GliovascularConnectivity
            self._circuit.edges[Population.GLIOVASCULAR] = GliovascularConnectivity(
                self._config["gliovascular"])
        return self._circuit.edges

    @cached_property
    def vasculature(self):
        """"Access to the vasculature objects."""
        if "vasculature" in self._config:
            filepath = self._config["vasculature"]["vasculature_file"]
            meshpath = self._config["vasculature"]["vasculature_mesh_file"]
            return Vasculature(filepath, meshpath)
        raise NGVError("No vasculature provided in the circuit config")

    @cached_property
    def atlases(self):
        """Returns atlases."""
        return _load_atlases(self._config["atlases"])
