import os
import logging

import numpy as np

from .connectivity_gliovascular import GliovascularConnectivity
from .connectivity_neuroglial import NeuroglialConnectivity
from .connectivity_synaptic import SynapticConnectivity

L = logging.getLogger(__name__)


class NGVConnectome(object):

    def __init__(self, ngv_config):

        self._snc = SynapticConnectivity(ngv_config.output_paths('synaptic_connectivity'))
        self._ngc = NeuroglialConnectivity(ngv_config.output_paths('neuroglial_connectivity'))
        self._gvc = GliovascularConnectivity(ngv_config.output_paths('gliovascular_connectivity'))

    def close(self):
        self._snc.close()
        self._ngc.close()
        self._gvc.close()

    ############### Astrocyte Stuff ###############

    def astrocyte_endfeet(self, astrocyte_index):
        return self._gvc.astrocyte.to_endfoot(astrocyte_index)

    def astrocyte_vasculature_segments(self, astrocyte_index):
        return self._gvc.astrocyte.to_vasculature_segment(astrocyte_index)

    def astrocyte_synapses(self, astrocyte_index):
        return self._ngc.astrocyte.to_synapse(astrocyte_index)

    ############### Endfoot Stuff ###############

    def endfoot_vasculature_segment(self, endfoot_index):
        return self._gvc.endfoot.to_vasculature_segment(endfoot_index)

    def endfoot_astrocyte(self, endfoot_index):
        return self._gvc.endfoot.to_astrocyte(endfoot_index) 

    ############### Synapse Stuff ###############

    def synapse_afferent_neuron(self, synapse_index):
        return self._snc.synapse.to_afferent_neuron(synapse_index)

    def synapse_astrocyte(self, synapse_index):
        return self._ngc.synapse.to_astrocyte(synapse_index)

    ############### Vasculature Segment Stuff ###############

    def vasculature_segment_endfoot(self, vasculature_segment_index):
        return self._gvc.vasculature_segment.to_endfoot(vasculature_segment_index)

    def vasculature_segment_astrocyte(self, vasculature_segment_index):
        return self._gvc.vasculature_segment.to_astrocyte(vasculature_segment_index)


    ############### Pairwise Combinations ###############

    def astrocyte_afferent_neurons(self, astrocyte_index):
        syn_idx = self.astrocyte_synapses(astrocyte_index)
        return set(self.synapse_afferent_neuron(index) for index in syn_idx)
