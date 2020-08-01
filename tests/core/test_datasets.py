import pytest
import numpy.testing as npt
import numpy as np

import archngv.core.datasets as tested
from archngv.exceptions import NGVError

from utils import get_data


class TestCellData:
    def setup(self):
        self.cells = tested.CellData(get_data("nodes.h5"))

    def test_positions(self):
        expected = np.array([[1., 2., 3.], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]])
        npt.assert_allclose(self.cells.positions(), expected)


class TestGliovascularConnectivity:
    def setup(self):
        self.gliovascular = tested.GliovascularConnectivity(get_data("gliovascular.h5"))

    def test_vasculature_surface_targets(self):
        expected = np.array([[0.11, 0.21, 0.31], [0.12, 0.22, 0.32], [0.13, 0.23, 0.33]])
        npt.assert_allclose(self.gliovascular.vasculature_surface_targets, expected.T)

    def test_astrocyte_endfeet(self):
        assert self.gliovascular.astrocyte_endfeet(0) == 2
        assert self.gliovascular.astrocyte_endfeet(1) == 1
        assert self.gliovascular.astrocyte_endfeet(2) == 0


class TestNeuronalConnectivity:
    def setup(self):
        self.neuronal = tested.NeuronalConnectivity(get_data("edges.h5"))

    def test_synapse_positions(self):
        expected = np.array([[1110.0, 1111.0, 1112.0, 1113.0],
                             [1120.0, 1121.0, 1122.0, 1123.0],
                             [1130.0, 1131.0, 1132.0, 1133.0]])
        npt.assert_allclose(self.neuronal.synapse_positions(), expected.T)

    def test_target_neurons(self):
        npt.assert_equal(self.neuronal.target_neurons(), [0, 1, 1, 1])

    def test_target_neuron_count(self):
        assert self.neuronal.target_neuron_count == 2

    def test_fail_positions(self):
        with pytest.raises(NGVError):
            tested.NeuronalConnectivity(get_data("edges_fail.h5")).synapse_positions()


class TestNeuroglialConnectivity:
    def setup(self):
        self.neuroglial = tested.NeuroglialConnectivity(get_data("neuroglial.h5"))

    def test_astrocyte_synapses(self):
        npt.assert_equal(self.neuroglial.astrocyte_synapses(0), [1])
        npt.assert_equal(self.neuroglial.astrocyte_synapses(1), [3])
        npt.assert_equal(self.neuroglial.astrocyte_synapses(2), [0, 1])

    def test_astrocyte_number_of_synapses(self):
        assert self.neuroglial.astrocyte_number_of_synapses(2) == 2
        assert self.neuroglial.astrocyte_number_of_synapses(1) == 1
        assert self.neuroglial.astrocyte_number_of_synapses(0) == 1

    def test_astrocyte_neurons(self):
        npt.assert_equal(self.neuroglial.astrocyte_neurons(0), [1])
        npt.assert_equal(self.neuroglial.astrocyte_neurons(1), [1])
        npt.assert_equal(self.neuroglial.astrocyte_neurons(2), [0, 1])


class TestGlialglialConnectivity:
    def setup(self):
        self.glialglial = tested.GlialglialConnectivity(get_data("glialglial.h5"))

    def test_astrocyte_astrocytes(self):
        npt.assert_equal(self.glialglial.astrocyte_astrocytes(0), [1])
        npt.assert_equal(self.glialglial.astrocyte_astrocytes(1), [])
        npt.assert_equal(self.glialglial.astrocyte_astrocytes(2), [0, 1])
        npt.assert_equal(self.glialglial.astrocyte_astrocytes(0, unique=False), [1, 1])
