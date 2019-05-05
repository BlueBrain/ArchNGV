import os
import h5py
import pytest
import numpy as np
from unittest.mock import Mock

from ..connectivity_neuroglial import NeuroglialConnectivity
from ...exporters.export_neuroglial_connectivity import export_neuroglial_connectivity

N_ASTROCYTES = 4
N_NEURONS = 8
N_SYNAPSES = 40




class MockNeuroglialConnectivity:

    def __init__(self):

        self.synapses_per_astrocyte = \
                np.split(np.arange(N_SYNAPSES, dtype=np.uintp), N_ASTROCYTES)

        self.neurons_per_astrocyte = \
                np.split(np.arange(N_NEURONS, dtype=np.uintp), N_ASTROCYTES)

        self.astrocyte = Mock(to_neuron = lambda astro_index: self.neurons_per_astrocyte[astro_index],
                              to_synapse = lambda astro_index: self.synapses_per_astrocyte[astro_index])

    @property
    def astrocytes_per_neuron(self):

        a_p_n = [[] for _ in range(N_NEURONS)]

        for astro_index, neurons in enumerate(self.neurons_per_astrocyte):
            for neuron in neurons:
                a_p_n[neuron].append(astro_index)
        return a_p_n

    @property
    def neuron_to_astrocyte(self):
        res = np.zeros(N_SYNAPSES, dtype=np.uintp)
        offset = 0
        for astrocyte_index, neurons in enumerate(self.neurons_per_astrocyte):
            n_neurons = len(neurons)
            res[offset: offset + n_neurons] = astrocyte_index
            offset += n_neurons
        return res

    def data_generator(self):
        return zip(self.synapses_per_astrocyte, self.neurons_per_astrocyte)


@pytest.fixture(scope='session')
def ng_conn_path(tmpdir_factory):

    directory_path = tmpdir_factory.getbasetemp()

    path = os.path.join(directory_path, 'synaptic_data.h5')
    return path


@pytest.fixture(scope='module')
def ng_conn_mock(ng_conn_path):

    mock_data = MockNeuroglialConnectivity()

    export_neuroglial_connectivity(mock_data.data_generator(),
                                     N_ASTROCYTES,
                                     N_SYNAPSES,
                                     N_NEURONS,
                                     ng_conn_path)
    return mock_data


@pytest.fixture(scope='module')
def ng_conn_data(ng_conn_path, ng_conn_mock): # ensure ng_conn_mock created first
    return NeuroglialConnectivity(ng_conn_path)


def test_number_of_astrocytes(ng_conn_data):
    assert ng_conn_data.n_astrocytes == N_ASTROCYTES


def test_astrocyte_to_neurons(ng_conn_data, ng_conn_mock):
    for astro_index in range(N_ASTROCYTES):
        assert np.all(ng_conn_data.astrocyte.to_neuron(astro_index) ==
                      ng_conn_mock.astrocyte.to_neuron(astro_index))

def test_astrocyte_to_synapses(ng_conn_data, ng_conn_mock):
    for astro_index in range(N_ASTROCYTES):
        assert np.all(ng_conn_data.astrocyte.to_synapse(astro_index) == \
                      ng_conn_mock.astrocyte.to_synapse(astro_index))

