import os
import h5py
import logging
import collections
from enum import Enum

import numpy as np

from collections import namedtuple
from itertools import groupby
from copy import deepcopy


from .common import H5ContextManager


L = logging.getLogger(__name__)


class NeuroglialConnectivity(H5ContextManager):


    def __init__(self, filepath):
        super(NeuroglialConnectivity, self).__init__(filepath)

        #self.neuron = self._NeuronEntry(self._fd)
        #self.synapse = self._SynapseEntry(self._fd)
        self.astrocyte = AstrocyteEntry(self._fd)

    @property
    def n_astrocytes(self):
        return len(self.astrocyte._offsets) - 1

    @property
    def n_neurons(self):
        return len(self.neuron._offsets) - 1

    @property
    def n_synapses(self):
        raise NotImplementedError


class NeuronEntry(object):

    def __init__(self, fd):

        self._offsets = fd['/Neuron/offsets']
        self._astrocyte = fd['/Neuron/astrocyte']

    def _offset_slice(self, neuron_index):
         return self._offsets[neuron_index], \
                self._offsets[neuron_index + 1]

    def to_astrocyte(self, neuron_index):
        beg, end = self._offset_slice(neuron_index)
        return self._astrocyte[beg: end]


class SynapseEntry(object):

    def __init__(self, fd):

        self._offset_t = \
        {
            'astrocyte': 0
        }

        self._offsets = fd['/Synapse/offsets']
        self._astrocyte = fd['/Synapse/astrocyte']

    def _offset_slice(self, synapse_index, offset_type):
         return self._offsets[neuron_index, offset_type], \
                self._offsets[neuron_index + 1, offset_type]

    def to_astrocyte(self, synapse_index):
        return self._connectivity[synapse_index, self._target_t['astrocyte']]


class AstrocyteEntry(object):

    def __init__(self, fd):

        self._offset_t = \
        {
            'synapse': 0,
            'neuron' : 1
        }

        self._offsets = fd['/Astrocyte/offsets']
        self._synapse = fd['/Astrocyte/synapse']
        self._neuron = fd['/Astrocyte/neuron']

    def _offset_slice(self, astrocyte_index, offset_type):
        return self._offsets[astrocyte_index, offset_type], \
               self._offsets[astrocyte_index + 1, offset_type]

    def to_synapse(self, astrocyte_index):
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['synapse'])
        return self._synapse[beg: end]

    def to_neuron(self, astrocyte_index):
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['neuron'])
        return self._neuron[beg: end]
