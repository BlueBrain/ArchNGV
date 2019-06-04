""" Container for NGV connectome """

import logging

from cached_property import cached_property

from .connectivity_gliovascular import GliovascularConnectivity
from .connectivity_neuroglial import NeuroglialConnectivity


L = logging.getLogger(__name__)


class NGVConnectome(object):
    """ Connectome container for accessing the NGV connectivities
    """
    def __init__(self, ngv_config):

        self._config = ngv_config

        self._connectivities = {'neuroglial': self._neuroglial,
                                'gliovascular': self._gliovascular,
                                'synaptic': self._synaptic}

    @property
    def connectivities(self):
        """ Available connectivities """
        return self._connectivities.keys()

    def __getitem__(self, connectivity_name):
        """ Get connectivity by name """
        return self._connectivities[connectivity_name]()

    def _gliovascular(self):
        """ Return gliovascular connectivity """
        filepath = self._config.output_paths('gliovascular_connectivity')
        return GliovascularConnectivity(filepath)

    @cached_property
    def gliovascular(self):
        """ Get gliovascular connectivity """
        return self._gliovascular()

    def _neuroglial(self):
        """ Get neuroglial connectivity """
        filepath = self._config.output_paths('neuroglial_connectivity')
        return NeuroglialConnectivity(filepath)

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

    def __enter__(self):
        """ Composition context manager """
        self.synaptic.__enter__()
        self.neuroglial.__enter__()
        self.gliovascular.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Called when with goes out of scope """
        self.close()

    def close(self):
        """ Close context managers """
        self.synaptic.close()
        self.neuroglial.close()
        self.gliovascular.close()

    def astrocyte_endfeet(self, astrocyte_index):
        """ Given an astrocyte index, return the endfeet indices """
        return self.gliovascular.astrocyte.to_endfoot(astrocyte_index)

    def astrocyte_vasculature_segments(self, astrocyte_index):
        """ Given an astrocyte index return the vasculature segment indices """
        return self.gliovascular.astrocyte.to_vasculature_segment(astrocyte_index)

    def astrocyte_synapses(self, astrocyte_index):
        """ Given an astrocyte index return the synapse indices """
        return self.neuroglial.astrocyte.to_synapse(astrocyte_index)

    def endfoot_vasculature_segment(self, endfoot_index):
        """ Given the endfoot index return the vasculature segment indices """
        return self.gliovascular.endfoot.to_vasculature_segment(endfoot_index)

    def endfoot_astrocyte(self, endfoot_index):
        """ Given an endfoot index return the respective astrocyte index """
        return self.gliovascular.endfoot.to_astrocyte(endfoot_index)

    def synapse_afferent_neuron(self, synapse_index):
        """ Given the synapse index return the afferent neuron index """
        return self.synaptic.synapse.to_afferent_neuron(synapse_index)

    def synapse_astrocyte(self, synapse_index):
        """ Given the synapse index return the respective astrocyte indices """
        return self.neuroglial.synapse.to_astrocyte(synapse_index)

    def vasculature_segment_endfoot(self, vasculature_segment_index):
        """ Given the vasculature segment index return the respective endfoot """
        return self.gliovascular.vasculature_segment.to_endfoot(vasculature_segment_index)

    def vasculature_segment_astrocyte(self, vasculature_segment_index):
        """ Given the vasculature segment index return the connecting astrocyte """
        return self.gliovascular.vasculature_segment.to_astrocyte(vasculature_segment_index)

    def astrocyte_afferent_neurons(self, astrocyte_index):
        """ Given the astrocyte index return the afferent neuron indices """
        syn_idx = self.astrocyte_synapses(astrocyte_index)
        return set(self.synapse_afferent_neuron(index) for index in syn_idx)
