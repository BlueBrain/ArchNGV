import filecmp
import logging
import os
from pathlib import Path

import jsondiff
import meshio
import pytest
import vascpy
from bluepysnap import Circuit
from numpy import testing as npt

from archngv.app.utils import load_json
from archngv.core.datasets import CellData
from archngv.spatial import BoundingBox

L = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.resolve() / "data"
BUILD_DIR = Path(__file__).parent.resolve() / "build"
EXPECTED_DIR = Path(__file__).parent.resolve() / "expected"

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


def test_surface_mesh():
    """Compare STP surface mesh"""
    import trimesh

    assert (
        filecmp.cmp(
            EXPECTED_DIR / "ngv_refined_tetrahedral_mesh.msh",
            BUILD_DIR / "ngv_refined_tetrahedral_mesh.msh",
        )
        == 1
    )

    assert (
        filecmp.cmp(
            EXPECTED_DIR / "ngv_prepared_tetrahedral_mesh.geo",
            BUILD_DIR / "ngv_prepared_tetrahedral_mesh.geo",
        )
        == 1
    )


def test_nodes_inside_mesh():
    """test that every node (cells soma and vasculature nodes)
    are included in the tetrahedral mesh
    """

    neuron_pos = CellData(DATA_DIR / "circuit/nodes.h5").positions()
    glial_pos = CellData(BUILD_DIR / "sonata/networks/nodes/astrocytes/nodes.h5").positions()
    vasc_pos = vascpy.PointVasculature.load_sonata(
        BUILD_DIR / "sonata/networks/nodes/vasculature/nodes.h5"
    ).points
    mesh_pos = meshio.read(BUILD_DIR / "ngv_refined_tetrahedral_mesh.msh").points

    bb = BoundingBox.from_points(mesh_pos)

    astro_mask = bb.points_inside(glial_pos)
    assert astro_mask.all()

    neuron_mask = bb.points_inside(neuron_pos)
    assert neuron_mask.all()

    vasculature_mask = bb.points_inside(vasc_pos)
    assert vasculature_mask.all()


def test_morphologies():
    """Compare synthesized hdf5 morphologies"""
    from morph_tool.morphio_diff import diff

    filenames = _filenames_verify_cardinality(
        BUILD_DIR / MORPHOLOGIES_DIR, EXPECTED_DIR / MORPHOLOGIES_DIR
    )

    for filename in filenames:
        diff_result = diff(
            BUILD_DIR / MORPHOLOGIES_DIR / filename,
            EXPECTED_DIR / MORPHOLOGIES_DIR / filename,
            atol=1e-4,
        )

        assert not diff_result, diff_result.info


def _h5_compare(actual_filepath, expected_filepath):
    import subprocess

    completed_process = subprocess.run(
        ["h5diff", "-v", "-c", "--delta=1e-4", actual_filepath, expected_filepath]
    )

    assert completed_process.returncode == 0


def _h5_compare_all(actual_dir, expected_dir):
    for filename in _filenames_verify_cardinality(actual_dir, expected_dir):
        _h5_compare(actual_dir / filename, expected_dir / filename)


def test_sonata_files():
    actual_files = sorted(Path(BUILD_DIR / "sonata/networks").rglob("*.h5"))
    expected_files = sorted(Path(EXPECTED_DIR / "sonata/networks").rglob("*.h5"))

    assert len(actual_files) > 0
    assert len(expected_files) > 0

    assert [p.relative_to(BUILD_DIR) for p in actual_files] == [
        p.relative_to(EXPECTED_DIR) for p in expected_files
    ]
    for actual, expected in zip(actual_files, expected_files):
        _h5_compare(actual, expected)


def test_root_files():
    """Files at the root level of build-expected"""
    _h5_compare_all(BUILD_DIR, EXPECTED_DIR)


def test_config():
    expected_sonata_config = {
        "manifest": {
            "$CIRCUIT_DIR": "../",
            "$BASE_DIR": "$CIRCUIT_DIR/build",
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
                    "nodes_file": "$BASE_DIR/sonata/networks/nodes/vasculature/nodes.h5",
                    "populations": {
                        "vasculature": {
                            "type": "vasculature",
                            "vasculature_file": f"{DATA_DIR}/atlas/vasculature.h5",
                            "vasculature_mesh": f"{DATA_DIR}/atlas/vasculature.obj",
                        },
                    },
                },
                {
                    "nodes_file": "$BASE_DIR/sonata/networks/nodes/astrocytes/nodes.h5",
                    "populations": {
                        "astrocytes": {
                            "type": "astrocyte",
                            "alternate_morphologies": {"h5v1": "$BASE_DIR/morphologies"},
                            "microdomains_file": "$BASE_DIR/microdomains.h5",
                        },
                    },
                },
            ],
            "edges": [
                {
                    "edges_file": f"{DATA_DIR}/circuit/edges.h5",
                    "populations": {
                        "All": {"type": "chemical"},
                    },
                },
                {
                    "edges_file": "$BASE_DIR/sonata/networks/edges/neuroglial/edges.h5",
                    "populations": {"neuroglial": {"type": "synapse_astrocyte"}},
                },
                {
                    "edges_file": "$BASE_DIR/sonata/networks/edges/glialglial/edges.h5",
                    "populations": {"glialglial": {"type": "glialglial"}},
                },
                {
                    "edges_file": "$BASE_DIR/sonata/networks/edges/gliovascular/edges.h5",
                    "populations": {
                        "gliovascular": {
                            "type": "endfoot",
                            "endfeet_meshes_file": "$BASE_DIR/endfeet_meshes.h5",
                        }
                    },
                },
            ],
        },
        "metadata": {"status": "partial"},
    }

    build_sonata_config = load_json(BUILD_DIR / "ngv_config.json")
    assert build_sonata_config == expected_sonata_config
    assert not jsondiff.diff(build_sonata_config, expected_sonata_config)
