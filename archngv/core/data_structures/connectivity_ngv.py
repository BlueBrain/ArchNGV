import os
import logging

import numpy as np

from .connectivity_gliovascular import GliovascularConnectivity
from .connectivity_neuroglial import NeuroglialConnectivity
from .connectivity_synaptic import SynapticConnectivity


L = logging.getLogger(__name__)


class NGVConnectome(object):

    def __init__(self, ngv_config):

        self.synaptic = SynapticConnectivity(ngv_config.output_paths('synaptic_connectivity'))
        self.neuroglial = NeuroglialConnectivity(ngv_config.output_paths('neuroglial_connectivity'))
        self.gliovascular = GliovascularConnectivity(ngv_config.output_paths('gliovascular_connectivity'))

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
        self.synaptic.close()
        self.neuroglial.close()
        self.gliovascular.close()

    @property
    def gliovascular(self):
        return self._gvc

    @property
    def neuroglial(self):
        return self._ngc

    @property
    def synaptic(self):
        return self._snc

    ############### Astrocyte Stuff ###############

    def astrocyte_endfeet(self, astrocyte_index):
        return self.gliovascular.astrocyte.to_endfoot(astrocyte_index)

    def astrocyte_vasculature_segments(self, astrocyte_index):
        return self.gliovascular.astrocyte.to_vasculature_segment(astrocyte_index)

    def astrocyte_synapses(self, astrocyte_index):
        return self.neuroglial.astrocyte.to_synapse(astrocyte_index)

    ############### Endfoot Stuff ###############

    def endfoot_vasculature_segment(self, endfoot_index):
        return self.gliovascular.endfoot.to_vasculature_segment(endfoot_index)

    def endfoot_astrocyte(self, endfoot_index):
        return self.gliovascular.endfoot.to_astrocyte(endfoot_index) 

    ############### Synapse Stuff ###############

    def synapse_afferent_neuron(self, synapse_index):
        return self.synaptic.synapse.to_afferent_neuron(synapse_index)

    def synapse_astrocyte(self, synapse_index):
        return self.neuroglial.synapse.to_astrocyte(synapse_index)

    ############### Vasculature Segment Stuff ###############

    def vasculature_segment_endfoot(self, vasculature_segment_index):
        return self.gliovascular.vasculature_segment.to_endfoot(vasculature_segment_index)

    def vasculature_segment_astrocyte(self, vasculature_segment_index):
        return self.gliovascular.vasculature_segment.to_astrocyte(vasculature_segment_index)


    ############### Pairwise Combinations ###############

    def astrocyte_afferent_neurons(self, astrocyte_index):
        syn_idx = self.astrocyte_synapses(astrocyte_index)
        return set(self.synapse_afferent_neuron(index) for index in syn_idx)
