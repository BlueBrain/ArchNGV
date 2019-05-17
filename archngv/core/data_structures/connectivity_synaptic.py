""" Container for connectivities between synapses and neurons """

import logging

import numpy as np

from .common import H5ContextManager


L = logging.getLogger(__name__)


class SynapticConnectivity(H5ContextManager):
    """
    Arguments:
        filepath:
            Absolute path to hdf5 file.
    Attributes:
        synapse:
            Synapse view allows accesing connectivity from the synapse
            to afferent neuron.
        afferent_neuron:
            Afferent neuron view allows accesing connectivity from the
            afferent neuron to the synapse.
    """
    def __init__(self, filepath):
        super(SynapticConnectivity, self).__init__(filepath)
        self.synapse = SynapseEntry(self._fd)
        self.afferent_neuron = AfferentNeuronEntry(self._fd)

    @property
    def n_neurons(self):
        """ Total number of neurons """
        return len(self.afferent_neuron)

    @property
    def n_synapses(self):
        """ Total number of synapses """
        return len(self.synapse)


class SynapseEntry(object):
    """ Synaptic point of view. Allows access to all its
    neighbors.
    """
    def __init__(self, fd):
        self._afferent_neuron = fd['/Synapse/Afferent Neuron']

    def __len__(self):
        """ Number of synapses """
        return len(self._afferent_neuron)

    def to_afferent_neuron(self, synapse_index):
        """ Neuron index for synapse index """
        return self._afferent_neuron[synapse_index]

    @property
    def to_afferent_neuron_map(self):
        """ Whole synapse to neuron indices dataset """
        return self._afferent_neuron


class AfferentNeuronEntry(object):
    """ Neuronal point of view """
    def __init__(self, fd):
        self._offsets = fd['/Afferent Neuron/offsets']

    def __len__(self):
        """ Number of neurons """
        return len(self._offsets) - 1

    def _offset_slice(self, neuron_index):
        """ For a given neuron index return the synapse index slice """
        return self._offsets[neuron_index], \
               self._offsets[neuron_index + 1]

    def to_synapse(self, neuron_index):
        """ Synapses indices for neuron index """
        beg, end = self._offset_slice(neuron_index)
        return np.arange(beg, end, dtype=np.uintp)
