from pathlib import Path

import pandas as pd
import numpy as np
import trimesh
from voxcell import VoxelData

import pytest
from numpy import testing as npt
from pandas.testing import assert_frame_equal
from mock import patch, PropertyMock

import archngv.core.circuit as test_module
from archngv.core.structures import Microdomains, Atlas
from archngv.core.datasets import Vasculature, MicrodomainTesselation, EndfootSurfaceMeshes

from archngv.exceptions import NGVError

from utils import get_data

TEST_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = Path(TEST_DIR / "data").resolve()


def test__find_config_location():
    assert test_module._find_config_location(
        TEST_DATA_DIR) == TEST_DATA_DIR / "build/ngv_config.json"
    assert test_module._find_config_location(
        get_data("circuit_config.json")) == TEST_DATA_DIR / "circuit_config.json"
    with pytest.raises(NGVError):
        test_module._find_config_location("missing")


class TestCircuit:
    def setup(self):
        self.circuit = test_module.NGVCircuit(TEST_DATA_DIR / "circuit_config.json")

    def test_config(self):
        assert (
                self.circuit.config['networks']["nodes"][0] ==
                {
                    "nodes_file": str(Path(TEST_DATA_DIR, "glia.h5").resolve()),
                    "node_types_file": None,
                }
        )

        assert (
                self.circuit.config['networks']["nodes"][1] ==
                {
                    "nodes_file": str(Path(TEST_DATA_DIR, "nodes.h5").resolve()),
                    "node_types_file": None,
                }
        )

    def test_populations(self):
        assert isinstance(self.circuit.nodes, dict)
        assert sorted(list(self.circuit.nodes)) == sorted(["default", "vasculature", "astrocytes"])
        assert isinstance(self.circuit.edges, dict)
        assert sorted(list(self.circuit.edges)) == sorted(
            ["default", "glialglial", "neuroglial", "gliovascular"])

    def test_population_instances(self):
        assert isinstance(self.circuit.nodes["default"], test_module.NGVNodes)
        assert self.circuit.nodes["default"] is self.circuit.neurons

        assert isinstance(self.circuit.nodes["astrocytes"], test_module.Astrocytes)
        assert self.circuit.nodes["astrocytes"] is self.circuit.astrocytes

        assert isinstance(self.circuit.nodes["vasculature"], test_module.Vasculature)
        assert self.circuit.nodes["vasculature"] is self.circuit.vasculature

        assert isinstance(self.circuit.edges["default"], test_module.NGVEdges)
        assert self.circuit.edges["default"] is self.circuit.neuronal_connectome

        assert isinstance(self.circuit.edges["glialglial"], test_module.GlialGlial)
        assert self.circuit.edges["glialglial"] is self.circuit.glialglial_connectome

        assert isinstance(self.circuit.edges["gliovascular"], test_module.GlioVascular)
        assert self.circuit.edges["gliovascular"] is self.circuit.gliovascular_connectome

        assert isinstance(self.circuit.edges["neuroglial"], test_module.NeuroGlial)
        assert self.circuit.edges["neuroglial"] is self.circuit.neuroglial_connectome

        assert isinstance(self.circuit.atlases, dict)
        assert isinstance(self.circuit.atlases["my_atlas"], Atlas)

    def test_neurons_morphologies(self):
        neurons = self.circuit.neurons
        assert neurons.morphology is neurons.morph
        assert neurons.morphology.get_filepath(0) == neurons.morph.get_filepath(0)
        npt.assert_allclose(neurons.morphology.get(0).points,
                            neurons.morph.get(0).points)
        assert str(neurons.morph.get_filepath(0)) == get_data('morphologies/morph-A.swc')
        actual = neurons.morph.get(0).points
        assert len(actual) == 13
        expected = [
            [0., 5., 0.],
            [2., 9., 0.]
        ]
        npt.assert_almost_equal(expected, actual[:2])

    def test_astrocytes_morphologies(self):
        astrocytes = self.circuit.astrocytes
        assert astrocytes.morphology is astrocytes.morph
        assert astrocytes.morphology.get_filepath(0) == astrocytes.morph.get_filepath(0)
        npt.assert_allclose(astrocytes.morphology.get(0).points, astrocytes.morph.get(0).points)
        assert astrocytes.morph.get_filepath(0) == get_data('morphologies-astro/morph-A.h5')
        # Using the same morph for the neurons and the astrocytes explains why we have the
        # same test as for neurons
        actual = astrocytes.morph.get(0).points
        assert len(actual) == 13
        expected = [
            [0., 5., 0.],
            [2., 9., 0.]
        ]
        npt.assert_almost_equal(expected, actual[:2])



    def test_vasculature_morphology(self):
        vasculature = self.circuit.vasculature
        assert vasculature.morphology is vasculature.morph
        assert isinstance(vasculature.morphology, Vasculature)

        npt.assert_allclose(vasculature.morphology.points, vasculature.morph.points)
        npt.assert_allclose(vasculature.morphology.points[:3], [[0., 0., 4650.],
                                                                [0., 0., 4665.],
                                                                [0., 0., 4680.]])

    def test_astrocyte_api(self):
        astrocytes = self.circuit.astrocytes
        assert isinstance(astrocytes.microdomains, Microdomains)
        assert isinstance(astrocytes.microdomains.tesselation, MicrodomainTesselation)
        assert isinstance(astrocytes.microdomains.overlapping, MicrodomainTesselation)

    def test_vasculature_api(self):
        vasculature = self.circuit.vasculature
        assert isinstance(vasculature.surface_mesh, trimesh.base.Trimesh)

    def test_gliovascular_api(self):
        gv = self.circuit.gliovascular_connectome
        assert gv.astrocytes is self.circuit.astrocytes
        assert gv.vasculature is self.circuit.vasculature
        assert isinstance(gv.surface_meshes, EndfootSurfaceMeshes)
        npt.assert_allclose(gv.surface_meshes.get('surface_area')[0], 119.36381)

        npt.assert_allclose(gv.astrocyte_endfeet(0), [2])
        npt.assert_allclose(gv.astrocyte_endfeet(1), [1])
        npt.assert_allclose(gv.astrocyte_endfeet(2), [0])

        npt.assert_allclose(gv.vasculature_endfeet(0), [0])
        npt.assert_allclose(gv.vasculature_endfeet(1), [1])
        npt.assert_allclose(gv.vasculature_endfeet(2), [2])

        npt.assert_equal(gv.connected_astrocytes(), [0, 1, 2])
        # astrocytes connected to a group of vasculature
        vgroup = {"section_id": 0}
        npt.assert_equal(gv.connected_astrocytes(vgroup), [1, 2])
        vgroup = {"end_diameter": (0.52, 0.9)}
        npt.assert_equal(gv.connected_astrocytes(vgroup), [0, 1])

        npt.assert_equal(gv.connected_vasculature(), [0, 1, 2])
        agroup = {"radius": (0, 2.5)}
        npt.assert_equal(gv.connected_vasculature(agroup), [1, 2])
        agroup = {"morphology": "morph-A"}
        npt.assert_equal(gv.connected_vasculature(agroup), [2])

        npt.assert_allclose(gv.vasculature_surface_targets([0, 1]).to_numpy(),
                            [[0.11, 0.12, 0.13], [0.21, 0.22, 0.23]])

        npt.assert_equal(gv.vasculature_sections_segments([0, 1]).to_numpy(),
                         [[0, 0, 0], [1, 0, 1]])


    def test_neuroglial_api(self):
        ng = self.circuit.neuroglial_connectome
        assert ng.astrocytes is self.circuit.astrocytes
        assert ng.neurons is self.circuit.neurons
        assert ng.synapses is self.circuit.neuronal_connectome

        npt.assert_equal(ng.astrocyte_synapses(0), [1])
        npt.assert_equal(ng.astrocyte_synapses(1), [3])
        npt.assert_equal(ng.astrocyte_synapses(2), [0, 1])

        data = {'afferent_center_x': [1110.0, 1111.0], 'syn_weight': [1.0, 1.0]}
        expected = pd.DataFrame(data=data, index=np.array([0, 1], dtype=np.int64))
        res = ng.astrocyte_synapses_properties(2, properties=['afferent_center_x', 'syn_weight'])
        assert_frame_equal(res, expected)

        data = {'x': [1111.0], 'y': [1121.0], 'z': [1131.0]}
        expected = pd.DataFrame(data=data, index=np.array([1], dtype=np.int64))
        res = ng.synapses.positions(ng.astrocyte_synapses(0), 'afferent', 'center')
        assert_frame_equal(res, expected)

        npt.assert_equal(ng.connected_neurons(), [0, 1])
        agroup = {"radius": (0, 2.5)}  # astro [0, 1]
        npt.assert_equal(ng.connected_neurons(agroup), [1])  # both connect to neuron 1

        npt.assert_equal(ng.connected_astrocytes(), [0, 1, 2])
        ngroup = {"layer": [1, 3]}  # neurons [0, 2]
        npt.assert_equal(ng.connected_astrocytes(ngroup), [2])

    def test_glialglial_api(self):
        gg = self.circuit.glialglial_connectome
        # should return the gap juction ids. That is, the corresponding edge ids
        npt.assert_equal(gg.astrocyte_gap_junctions(0), [0, 1, 2])
        npt.assert_equal(gg.astrocyte_gap_junctions(1), [1, 2, 3])
        npt.assert_equal(gg.astrocyte_gap_junctions(2), [0, 3])

        npt.assert_equal(gg.astrocyte_astrocytes(0), [1, 2])
        npt.assert_equal(gg.astrocyte_astrocytes(1), [0, 2])
        npt.assert_equal(gg.astrocyte_astrocytes(2), [0, 1])

        # accessing data from touches
        gap_junctions = gg.astrocyte_gap_junctions(0)
        prop = gg.properties(gap_junctions, properties=['spine_length', '@source_node',
                                                        '@target_node'])
        npt.assert_allclose(prop['spine_length'], [0.0, 0.1, 0.2])
        npt.assert_allclose(prop['@source_node'], [2, 0, 0])
        npt.assert_allclose(prop['@target_node'], [0, 1, 1])

    def test_atlas(self):
        assert isinstance(self.circuit.atlases["my_atlas"].get_atlas(), VoxelData)

    def test_repr(self):
        assert str(self.circuit) == f'<NGVCircuit : {TEST_DATA_DIR}/circuit_config.json>'

    def test_same_population(self):
        with patch('bluepysnap.nodes.NodeStorage.population_names',
                   new_callable=PropertyMock) as mock:
            mock.return_value = ["default", "default"]
            with pytest.raises(NGVError):
                test_module.NGVCircuit(TEST_DATA_DIR / "circuit_config.json").nodes

    def test_failing_get_population(self):
        with pytest.raises(NGVError):
            self.circuit._get_population("cells", "unknown")
