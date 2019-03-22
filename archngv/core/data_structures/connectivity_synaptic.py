import os
import logging

import numpy as np

from .common import H5ContextManager


L = logging.getLogger(__name__)


class SynapticConnectivity(H5ContextManager):


    def __init__(self, filepath):
        super(SynapticConnectivity, self).__init__(filepath)

        self.synapse = SynapseEntry(self._fd)
        self.afferent_neuron = AfferentNeuronEntry(self._fd)

    @property
    def n_neurons(self):
        return len(self.afferent_neuron._offsets) - 1

    @property
    def n_synapses(self):
        return len(self.synapse._afferent_neuron)


class SynapseEntry(object):

    def __init__(self, fd):
        self._afferent_neuron = fd['/Synapse/Afferent Neuron']

    def to_afferent_neuron(self, synapse_index):
        return self._afferent_neuron[synapse_index]

    @property
    def to_afferent_neuron_map(self):
        return self._afferent_neuron


class AfferentNeuronEntry(object):

    def __init__(self, fd):

        self._offsets = fd['/Afferent Neuron/offsets']

    def _offset_slice(self, neuron_index):
        return self._offsets[neuron_index], \
               self._offsets[neuron_index + 1]

    def to_synapse(self, neuron_index):
        beg, end = self._offset_slice(neuron_index)
        return np.arange(beg, end, dtype=np.uintp)
