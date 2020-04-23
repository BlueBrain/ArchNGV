from pathlib import Path

import pytest
from mock import Mock

from bluepysnap.nodes import NodePopulation
from bluepysnap.edges import EdgePopulation
from voxcell.voxel_data import VoxelData

import archngv.core.snapcircuit as test_module
from archngv.core.structures import (Endfeetome, Microdomains,
                                         Vasculature, Atlas)
from archngv.core.connectivities import GliovascularConnectivity
from archngv.core.datasets import MicrodomainTesselation
from archngv.core.datasets import EndfeetAreas
from archngv.core.datasets import GliovascularData
from archngv.core.datasets import Vasculature as VasculatureMorphology

from archngv.exceptions import NGVError


TEST_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = Path(TEST_DIR / "data").resolve()


def test__add_astrocytes_information():
    node_pop = Mock()
    config = {
        "microdomains_file": "microdomains_file",
        "microdomains_overlapping_file": "microdomains_overlapping_file",
        "endfeet_file": "endfeet_file",
        "endfeet_data_file": "endfeet_data_file",
        "morphologies_dir": "somewhere"
    }

    test_module._add_astrocytes_information(node_pop, config)
    assert isinstance(node_pop.microdomains, Microdomains)
    assert isinstance(node_pop.endfeetome, Endfeetome)

    assert node_pop.microdomains._microdomain_path == "microdomains_file"
    assert node_pop.microdomains._overlaping_path == "microdomains_overlapping_file"
    assert node_pop.endfeetome._areas_path == "endfeet_file"
    assert node_pop.endfeetome._data_path == "endfeet_data_file"
    assert node_pop.morph._morph_dir == "somewhere"


def test___load_atlases():
    config = {
        "my_atlas_1": "atlas_1.nrrd",
        "my_atlas_2": "atlas_2.nrrd"
    }
    atlases = test_module._load_atlases(config)
    assert isinstance(atlases, dict)
    assert sorted(list(atlases)) == ["my_atlas_1", "my_atlas_2"]
    assert isinstance(atlases["my_atlas_1"], Atlas)
    assert atlases["my_atlas_1"]._name == "my_atlas_1"
    assert atlases["my_atlas_1"]._filepath == "atlas_1.nrrd"


def test_all():
    circuit = test_module.NGVSnapCircuit(TEST_DATA_DIR / "circuit_config.json")
    assert(
        circuit._config["nodes"][0] ==
        {
            "nodes_file": str(Path(TEST_DATA_DIR, "glia.h5").resolve()),
            "node_types_file": None,
        }
    )

    assert(
        circuit._config["nodes"][1] ==
        {
            "nodes_file": str(Path(TEST_DATA_DIR, "nodes.h5").resolve()),
            "node_types_file": None,
        }
    )
    assert list(circuit._config) == ["nodes", "edges", "vasculature", "astrocytes",
                                     "gliovascular", "atlases"]

    assert isinstance(circuit.nodes, dict)
    assert sorted(list(circuit.nodes)) == sorted(["default", "default2", "astrocytes"])
    assert isinstance(circuit.edges, dict)
    assert sorted(list(circuit.edges)) == sorted(["default", "glialglial", "neuroglial", "gliovascular"])

    assert isinstance(circuit.nodes["default"], NodePopulation)
    assert isinstance(circuit.nodes["default2"], NodePopulation)
    assert isinstance(circuit.nodes["astrocytes"], NodePopulation)
    assert circuit.nodes["astrocytes"].microdomains._microdomain_path == str(Path(TEST_DATA_DIR, "microdomains.h5"))
    assert isinstance(circuit.nodes["astrocytes"].microdomains.tesselation, MicrodomainTesselation)
    assert isinstance(circuit.nodes["astrocytes"].microdomains.overlapping, MicrodomainTesselation)

    assert isinstance(circuit.nodes["astrocytes"].endfeetome.areas, EndfeetAreas)
    assert isinstance(circuit.nodes["astrocytes"].endfeetome.targets, GliovascularData)

    assert circuit.nodes["astrocytes"].microdomains._microdomain_path == str(Path(TEST_DATA_DIR, "microdomains.h5"))

    assert isinstance(circuit.edges["default"], EdgePopulation)
    assert isinstance(circuit.edges["glialglial"], EdgePopulation)
    assert isinstance(circuit.edges["neuroglial"], EdgePopulation)
    assert isinstance(circuit.edges["gliovascular"], GliovascularConnectivity)

    assert isinstance(circuit.vasculature, Vasculature)
    assert circuit.vasculature._vasculature_mesh_path == str(Path(TEST_DATA_DIR, "vasculature_mesh.obj"))
    assert len(circuit.vasculature.mesh.vertices) == 4
    assert circuit.vasculature._vasculature_path == str(Path(TEST_DATA_DIR, "vasculature.h5"))
    assert isinstance(circuit.vasculature.morphology, VasculatureMorphology)

    assert isinstance(circuit.atlases, dict)
    assert isinstance(circuit.atlases["my_atlas"], Atlas)
    assert isinstance(circuit.atlases["my_atlas"].get_atlas(), VoxelData)


def test_raise_vasculature():
    circuit = test_module.NGVSnapCircuit(TEST_DATA_DIR / "circuit_config.json")
    del circuit._config["vasculature"]
    with pytest.raises(NGVError):
        circuit.vasculature
