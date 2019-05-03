import os
import h5py
import pytest
import numpy as np

from ..data_synaptic import SynapticData


N_SYNAPSES = 7


class MockSynapticData(object):

    synapse_coordinates = np.random.random((N_SYNAPSES, 3))

    n_synapses = N_SYNAPSES

    def export(self, filepath):
        with h5py.File(filepath, 'w') as fd:
            fd.create_dataset('synapse_coordinates', data=self.synapse_coordinates)


@pytest.fixture(scope='session')
def syn_path(tmpdir_factory):

    directory_path = tmpdir_factory.getbasetemp()

    path = os.path.join(directory_path, 'synaptic_data.h5')
    return path


@pytest.fixture(scope='module')
def syn_mock(syn_path):

    mock_data = MockSynapticData()
    mock_data.export(syn_path)

    return mock_data


@pytest.fixture(scope='module')
def syn_data(syn_path, syn_mock):
    return SynapticData(syn_path)


def test_n_synapses(syn_data, syn_mock):
    assert syn_data.n_synapses == syn_mock.n_synapses


def test_synapse_coordinates(syn_data, syn_mock):
    assert np.allclose(syn_data.synapse_coordinates[:], syn_mock.synapse_coordinates)

