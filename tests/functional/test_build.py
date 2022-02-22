import logging
import os
from pathlib import Path

import pytest
import jsondiff
from numpy import testing as npt

from archngv.app.utils import load_json

L = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.resolve() / "data"
BUILD_DIR = Path(__file__).parent.resolve() / "build"
EXPECTED_DIR = Path(__file__).parent.resolve() / "expected"

SONATA_DIR = "./sonata"
MORPHOLOGIES_DIR = "./morphologies"
MICRODOMAINS_DIR = "./microdomains"


def _get_h5_files(directory):

    filenames = os.listdir(directory)
    filenames = filter(lambda s: s.endswith(".h5"), filenames)

    return sorted(filenames)


def _filenames_verify_cardinality(actual_directory, expected_directory):
    """Return the expected filenames and check if the produced filenames
    are identical in number and names.
    """
    actual_filenames = _get_h5_files(actual_directory)
    desired_filenames = _get_h5_files(expected_directory)

    npt.assert_equal(
        actual_filenames,
        desired_filenames,
        err_msg=(
            f"Differing output filenames:\n"
            f"Actual  : {sorted(actual_filenames)}\n"
            f"Expected: {sorted(desired_filenames)}"
        ),
    )

    return desired_filenames


def test_morphologies():
    """Compare synthesized hdf5 morphologies"""
    from morph_tool.morphio_diff import diff

    filenames = _filenames_verify_cardinality(
        BUILD_DIR / MORPHOLOGIES_DIR, EXPECTED_DIR / MORPHOLOGIES_DIR
    )

    for filename in filenames:

        diff_result = diff(
            BUILD_DIR / MORPHOLOGIES_DIR / filename, EXPECTED_DIR / MORPHOLOGIES_DIR / filename
        )

        assert not diff_result, diff_result.info


def _h5_compare(actual_filepath, expected_filepath):

    import subprocess

    completed_process = subprocess.run(
        ["h5diff", "-v", "-c", "--delta=1e-6", actual_filepath, expected_filepath]
    )

    assert completed_process.returncode == 0


def _h5_compare_all(actual_dir, expected_dir):
    for filename in _filenames_verify_cardinality(actual_dir, expected_dir):
        _h5_compare(actual_dir / filename, expected_dir / filename)


def test_sonata_files():

    _h5_compare_all(BUILD_DIR / SONATA_DIR / "nodes", EXPECTED_DIR / SONATA_DIR / "nodes")

    _h5_compare_all(BUILD_DIR / SONATA_DIR / "edges", EXPECTED_DIR / SONATA_DIR / "edges")


def test_microdomain_files():

    _h5_compare_all(BUILD_DIR / MICRODOMAINS_DIR, EXPECTED_DIR / MICRODOMAINS_DIR)


def test_root_files():
    """Files at the root level of build-expected"""
    _h5_compare_all(BUILD_DIR, EXPECTED_DIR)


def test_config():

    expected_sonata_config = {
        "manifest": {
            "$CIRCUIT_DIR": "../",
            "$BUILD_DIR": "$CIRCUIT_DIR/build",
            "$COMPONENT_DIR": "$BUILD_DIR",
            "$NETWORK_DIR": "$BUILD_DIR",
        },
        "networks": {
            "nodes": [
                {
                    "nodes_file": f"{DATA_DIR}/circuit/nodes.h5",
                    "populations": {
                        "All": {
                            "type": "biophysical",
                            "morphologies_dir": f"{DATA_DIR}/circuit/morphologies",
                        },
                    },
                },
                {
                    "nodes_file": "$NETWORK_DIR/sonata/nodes/vasculature.h5",
                    "populations": {
                        "vasculature": {
                            "type": "vasculature",
                            "vasculature_file": f"{DATA_DIR}/atlas/vasculature.h5",
                            "vasculature_mesh": f"{DATA_DIR}/atlas/vasculature.obj",
                        },
                    },
                },
                {
                    "nodes_file": "$NETWORK_DIR/sonata/nodes/glia.h5",
                    "populations": {
                        "astrocytes": {
                            "type": "protoplasmic_astrocytes",
                            "alternate_morphologies": {"h5v1": "$BUILD_DIR/morphologies"},
                            "microdomains_file": "$BUILD_DIR/microdomains/microdomains.h5",
                            "microdomains_overlapping_file": "$BUILD_DIR/microdomains/overlapping_microdomains.h5",
                        },
                    },
                },
            ],
            "edges": [
                {
                    "edges_file": f"{DATA_DIR}/circuit/edges.h5",
                    "populations": {
                        "All": {"type": "neuronal"},
                    },
                },
                {
                    "edges_file": "$NETWORK_DIR/sonata/edges/neuroglial.h5",
                    "populations": {"neuroglial": {"type": "neuroglial"}},
                },
                {
                    "edges_file": "$NETWORK_DIR/sonata/edges/glialglial.h5",
                    "populations": {"glialglial": {"type": "glialglial"}},
                },
                {
                    "edges_file": "$NETWORK_DIR/sonata/edges/gliovascular.h5",
                    "populations": {
                        "gliovascular": {
                            "type": "gliovascular",
                            "endfeet_areas": "$BUILD_DIR/endfeet_areas.h5",
                        }
                    },
                },
            ],
        },
        "atlases": {
            "intensity": f"{DATA_DIR}/atlas/[density]astrocytes.nrrd",
            "brain_regions": f"{DATA_DIR}/atlas/brain_regions.nrrd",
        },
    }

    build_sonata_config = load_json(BUILD_DIR / "ngv_config.json")
    assert not jsondiff.diff(build_sonata_config, expected_sonata_config)
