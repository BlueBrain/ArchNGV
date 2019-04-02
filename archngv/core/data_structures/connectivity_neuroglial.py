""" Data structures for connectivity between neurons and astrocytes
"""
from .common import H5ContextManager


class NeuroglialConnectivity(H5ContextManager):
    """
    Attributes:
        astrocyte: AtrocyteEntry
            Astrocyte point of view. Allow access of connections
            to synapses and neurons.
    """
    def __init__(self, filepath):
        super(NeuroglialConnectivity, self).__init__(filepath)

        #self.neuron = self._NeuronEntry(self._fd)
        #self.synapse = self._SynapseEntry(self._fd)
        self.astrocyte = AstrocyteEntry(self._fd)

    @property
    def n_astrocytes(self):
        """ Number of astrocytes"""
        return len(self.astrocyte)

    @property
    def n_neurons(self):
        """ Number of neurons """
        return len(self.neuron)

    @property
    def n_synapses(self):
        """ Number of synapses """
        raise NotImplementedError


class NeuronEntry(object):

    """ Astrocytic point of view. Allows access to all its
    neighbors.

    Attributes:
        offsets : hdf5 Dataset[init, (M + 1, 1)]
            The connectivity corresponding to the i-th
            astrocyte can be accesed as
            connectivity[offsets[i]: offsets[i + 1]].
            Note that for M astrocytes there are M + 1 rows
            as the end offest of the last astrocyte is contained
            as well. This is different than the usual h5v1 spec where
            it is left to the user to extract the last section from the
            number of points.
    """
    def __init__(self, fd):

        self._offsets = fd['/Neuron/offsets']
        self._astrocyte = fd['/Neuron/astrocyte']

    def __len__(self):
        """ Size """
        return len(self._offsets) - 1

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

    def __len__(self):
        """ Size """
        return len(self._offsets) - 1

    def _offset_slice(self, astrocyte_index, offset_type):
        return self._offsets[astrocyte_index, offset_type], \
               self._offsets[astrocyte_index + 1, offset_type]

    def to_synapse(self, astrocyte_index):
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['synapse'])
        return self._synapse[beg: end]

    def to_neuron(self, astrocyte_index):
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['neuron'])
        return self._neuron[beg: end]
