""" High level Circuit container. All NGV data can be accesed from here. """
from cached_property import cached_property
from archngv.core.config import NGVConfig


class NGVCircuit:
    """ Encapsulates all the information concerning an NGV
    circuit.

    Attributes:
        config: NGVConfig
    """
    def __init__(self, ngv_config):
        self._config = NGVConfig._resolve_config(ngv_config)

    def __repr__(self):
        """ Representation of circuit """
        return '<NGVCircuit @ {}>'.format(self.config.parent_directory)

    @property
    def config(self):
        """ NGV Config """
        return self._config

    @cached_property
    def data(self):
        """ Circuit data object """
        return NGVData(self.config)

    @cached_property
    def connectome(self):
        """ Circuit connectome object """
        return NGVConnectome(self.config)


class NGVData:
    """ Composition of all the ngv rich data structures
    """
    def __init__(self, ngv_config):
        self._config = ngv_config

    @cached_property
    def astrocytes(self):
        """ Returns cell data """
        from archngv.core.datasets import CellData
        return CellData(self._config)

    @cached_property
    def neurons(self):
        """ Returns neurons population """
        from bluepy.v2 import Circuit
        path = self._config.input_paths('microcircuit_path')
        return Circuit(path + '/CircuitConfig').cells

    @cached_property
    def synapses(self):
        """ Returns synaptic data """
        from archngv.core.datasets import SynapticData
        path = self._config.input_paths('synaptic_data')
        return SynapticData(path)

    @cached_property
    def vasculature(self):
        """ Returns vasculature object """
        from archngv.core.datasets import Vasculature
        return Vasculature.load(self._config.input_paths('vasculature'))

    @cached_property
    def vasculature_mesh(self):
        """ Returns vasculature mesh object """
        import trimesh
        return trimesh.load(self._config.input_paths('vasculature_mesh'))

    @cached_property
    def microdomains(self):
        """ Returns microdomain tesselation """
        from archngv.core.datasets import MicrodomainTesselation
        path = self._config.output_paths('microdomains')
        return MicrodomainTesselation(path)

    @cached_property
    def overlapping_microdomains(self):
        """ Returns overlapping microdomains """
        from archngv.core.datasets import MicrodomainTesselation
        path = self._config.output_paths('overlapping_microdomains')
        return MicrodomainTesselation(path)

    @cached_property
    def endfeetome(self):
        """ Returns endfeet areas data """
        from archngv.core.structures import Endfeetome
        areas_path = self._config.output_paths('endfeet_areas')
        gv_data_path = self._config.output_paths('gliovascular_data')
        return Endfeetome(areas_path, gv_data_path)

    @cached_property
    def voxelized_intensity(self):
        """ Returns atlas voxelized intensity """
        from voxcell import VoxelData
        path = self._config.input_paths('voxelized_intensity')
        return VoxelData.load_nrrd(path)


class NGVConnectome:
    """ Connectome container for accessing the NGV connectivities
    """
    def __init__(self, ngv_config):

        self._config = ngv_config

        self._connectivities = {'neuroglial': self._neuroglial,
                                'gliovascular': self._gliovascular,
                                'synaptic': self._synaptic,
                                'glial': self._glial}

    @property
    def connectivities(self):
        """ Available connectivities """
        return self._connectivities.keys()

    def __getitem__(self, connectivity_name):
        """ Get connectivity by name """
        return self._connectivities[connectivity_name]()

    def _gliovascular(self):
        """ Return gliovascular connectivity """
        from archngv.core.connectivities import GliovascularConnectivity
        filepath = self._config.output_paths('gliovascular_connectivity')
        return GliovascularConnectivity(filepath)

    @cached_property
    def gliovascular(self):
        """ Get gliovascular connectivity """
        return self._gliovascular()

    def _neuroglial(self):
        """ Get neuroglial connectivity """
        from archngv.core.connectivities import NeuroglialConnectivity
        return NeuroglialConnectivity(self._config.output_paths('neuroglial_connectivity'))

    @cached_property
    def neuroglial(self):
        """ Get neuroglial connectivity """
        return self._neuroglial()

    def _synaptic(self):
        """ Get synaptic connectivity """
        raise NotImplementedError

    @cached_property
    def synaptic(self):
        """ Get synaptic connectivity """
        return self._synaptic()

    def _glial(self):
        """ Get glial connectivity """
        from archngv.core.connectivities import GlialglialConnectivity
        return GlialglialConnectivity(self._config.output_paths('glialglial_connectivity'))

    @cached_property
    def glial(self):
        """ Get glial connectivity """
        return self._glial()

    def astrocyte_endfeet(self, astrocyte_index):
        """ Given an astrocyte index, return the endfeet indices """
        return self.gliovascular.astrocyte.to_endfoot(astrocyte_index)

    def astrocyte_synapses(self, astrocyte_index):
        """ Given an astrocyte index return the synapse indices """
        return self.neuroglial.astrocyte_synapses(astrocyte_index)

    def endfoot_vasculature_segment(self, endfoot_index):
        """ Given the endfoot index return the vasculature segment indices """
        return self.gliovascular.endfoot.to_vasculature_segment(endfoot_index)

    def endfoot_astrocyte(self, endfoot_index):
        """ Given an endfoot index return the respective astrocyte index """
        return self.gliovascular.endfoot.to_astrocyte(endfoot_index)

    def synapse_afferent_neuron(self, synapse_index):
        """ Given the synapse index return the afferent neuron index """
        return self.synaptic.synapse.to_afferent_neuron(synapse_index)

    def astrocyte_afferent_neurons(self, astrocyte_index):
        """ Given the astrocyte index return the afferent neuron indices """
        return self.neuroglial.astrocyte_neurons(astrocyte_index)
