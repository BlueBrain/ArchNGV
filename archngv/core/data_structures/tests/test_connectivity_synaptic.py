import os
import h5py
import pytest
import numpy as np
from unittest.mock import Mock

from ..connectivity_synaptic import SynapticConnectivity


N_NEURONS = 8
N_SYNAPSES = 40


class MockSynapticConnectivity(object):

    def __init__(self):

        self.synapses_per_neuron = \
             np.split(np.arange(N_SYNAPSES, dtype=np.uintp), N_NEURONS)
    
        self.synapse_to_neuron_connectivity = \
            [nrn_index for nrn_index, nrn_synapses in enumerate(self.synapses_per_neuron) for _ in nrn_synapses]

        self.afferent_neuron = \
            Mock(to_synapse=lambda nrn_index: self.synapses_per_neuron[nrn_index])

        self.synapse = \
            Mock(to_afferent_neuron=lambda sn_index: self.synapse_to_neuron_connectivity[sn_index])

    def export(self, filepath):
        with h5py.File(filepath, 'w') as fd_conn:

            # Synapse Point of view Group

            synapse_group = fd_conn.create_group('Synapse')
            dset_afferent_neuron = synapse_group.create_dataset('Afferent Neuron', data=self.synapse_to_neuron_connectivity)

            # Afferent Neuron Point of view Group

            afferent_neuron_group = fd_conn.create_group('Afferent Neuron')

            dset_afferent_neuron_offsets = \
            afferent_neuron_group.create_dataset('offsets',
                                                 shape=(N_NEURONS + 1,),
                                                 dtype=np.float32, chunks=None)
            dset_afferent_neuron_offsets[0] = 0
            dset_afferent_neuron_offsets.attrs['column_names'] = np.array(['Synapse'], dtype=h5py.special_dtype(vlen=str))

            offset = 0
            for neuron_index, synapses in enumerate(self.synapses_per_neuron):

                n_synapses =  len(synapses)
                dset_afferent_neuron[offset: offset + n_synapses] = neuron_index

                offset += n_synapses
                dset_afferent_neuron_offsets[neuron_index + 1] = offset


@pytest.fixture(scope='session')
def sn_conn_path(tmpdir_factory):

    directory_path = tmpdir_factory.getbasetemp()
    path = os.path.join(directory_path, 'synaptic_connectivity.h5')
    return path


@pytest.fixture(scope='module')
def sn_conn_mock(sn_conn_path):

    mock_data = MockSynapticConnectivity()
    mock_data.export(sn_conn_path)
    return mock_data


@pytest.fixture(scope='module')
def sn_conn_data(sn_conn_path, sn_conn_mock): # ensure sn_conn_mock created first
    return SynapticConnectivity(sn_conn_path)


def test_afferent_neuron_to_synapse(sn_conn_data, sn_conn_mock):
    for neuron_index in range(N_NEURONS):
        assert np.allclose(sn_conn_data.afferent_neuron.to_synapse(neuron_index),
                           sn_conn_mock.afferent_neuron.to_synapse(neuron_index))


def test_synapse_to_afferent_neuron(sn_conn_data, sn_conn_mock):
    for synapse_index in range(N_SYNAPSES):
        assert np.all(sn_conn_data.synapse.to_afferent_neuron(synapse_index) == \
                      sn_conn_mock.synapse.to_afferent_neuron(synapse_index))
