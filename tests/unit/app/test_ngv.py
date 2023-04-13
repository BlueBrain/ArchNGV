import shutil
import tempfile
import traceback
from pathlib import Path

import click.testing
import numpy as np
import pandas as pd
import pytest
import voxcell

from archngv.app import __main__ as main
from archngv.app import ngv as tested
from archngv.app.utils import load_yaml, write_yaml

DATA_DIR = Path(__file__).resolve().parent / "data"
BUILD_DIR = DATA_DIR / "frozen-build"
BIONAME_DIR = DATA_DIR / "bioname"
EXTERNAL_DIR = DATA_DIR / "external"

FIN_SONATA_DIR = BUILD_DIR / "sonata"
TMP_SONATA_DIR = BUILD_DIR / "sonata.tmp"


def assert_cli_run(cli, cmd_list):
    runner = click.testing.CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, [str(p) for p in cmd_list])
        assert result.exit_code == 0, "".join(traceback.format_exception(*result.exc_info))


def test_ngv_config():
    assert_cli_run(
        tested.ngv_config,
        [
            "--bioname",
            BIONAME_DIR,
            "--output",
            "ngv_config.json",
        ],
    )


def test_assign_emodels():
    assert_cli_run(
        tested.assign_emodels,
        [
            "--input",
            TMP_SONATA_DIR / "nodes/glia.somata.h5",
            "--hoc",
            "template-filename",
            "--output",
            "output_nodes.h5",
        ],
    )


def test_cell_placement():
    # TODO: Create a click sonata file type so that if sth else is passed to throw
    # meaningful error.

    assert_cli_run(
        tested.cell_placement,
        [
            "--config",
            BIONAME_DIR / "MANIFEST.yaml",
            "--atlas",
            EXTERNAL_DIR / "atlas",
            "--atlas-cache",
            ".atlas",
            "--vasculature",
            FIN_SONATA_DIR / "nodes/vasculature.h5",
            "--seed",
            0,
            "--population-name",
            "astrocytes",
            "--output",
            "output_nodes.h5",
        ],
    )


def test_cell_placement__with_region_specified():
    with tempfile.TemporaryDirectory() as tdir:
        edited_manifest = Path(tdir, "MANIFEST.yaml")

        shutil.copyfile(BIONAME_DIR / "MANIFEST.yaml", Path(tdir, "MANIFEST.yaml"))
        data = load_yaml(edited_manifest)
        data["ngv"]["common"]["region"] = "H"
        write_yaml(edited_manifest, data)

        assert_cli_run(
            tested.cell_placement,
            [
                "--config",
                str(edited_manifest),
                "--atlas",
                EXTERNAL_DIR / "atlas",
                "--atlas-cache",
                ".atlas",
                "--vasculature",
                FIN_SONATA_DIR / "nodes/vasculature.h5",
                "--seed",
                0,
                "--population-name",
                "astrocytes",
                "--output",
                "output_nodes.h5",
            ],
        )


def test_finalize_astrocytes():
    assert_cli_run(
        tested.finalize_astrocytes,
        [
            "--somata-file",
            TMP_SONATA_DIR / "nodes/glia.somata.h5",
            "--emodels-file",
            TMP_SONATA_DIR / "nodes/glia.emodels.h5",
            "--output",
            "glia.h5",
        ],
    )


def test_microdomains():
    assert_cli_run(
        tested.build_microdomains,
        [
            "--config",
            BIONAME_DIR / "MANIFEST.yaml",
            "--astrocytes",
            FIN_SONATA_DIR / "nodes/glia.h5",
            "--atlas",
            EXTERNAL_DIR / "atlas",
            "--atlas-cache",
            ".atlas",
            "--seed",
            0,
            "--output-file-path",
            "microdomains.h5",
        ],
    )


def test_gliovascular_connectivity():
    assert_cli_run(
        tested.gliovascular_connectivity,
        [
            "--config",
            BIONAME_DIR / "MANIFEST.yaml",
            "--astrocytes",
            FIN_SONATA_DIR / "nodes/glia.h5",
            "--microdomains",
            BUILD_DIR / "microdomains.h5",
            "--vasculature",
            FIN_SONATA_DIR / "nodes/vasculature.h5",
            "--seed",
            0,
            "--population-name",
            "gliovascular",
            "--output",
            "gliovascular.h5",
        ],
    )


def test_gliovascular_finalize():
    assert_cli_run(
        tested.attach_endfeet_info_to_gliovascular_connectivity,
        [
            "--input-file",
            TMP_SONATA_DIR / "edges/gliovascular.connectivity.h5",
            "--output-file",
            "gliovascular.h5",
            "--astrocytes",
            FIN_SONATA_DIR / "nodes/glia.h5",
            "--endfeet-meshes-path",
            BUILD_DIR / "endfeet_meshes.h5",
            "--vasculature-sonata",
            FIN_SONATA_DIR / "nodes/vasculature.h5",
            "--morph-dir",
            BUILD_DIR / "morphologies",
        ],
    )


def test_neuroglial_connectivity():
    assert_cli_run(
        tested.neuroglial_connectivity,
        [
            "--neurons-path",
            EXTERNAL_DIR / "circuit/nodes.h5",
            "--astrocytes-path",
            FIN_SONATA_DIR / "nodes/glia.h5",
            "--microdomains-path",
            BUILD_DIR / "microdomains.h5",
            "--neuronal-connectivity-path",
            EXTERNAL_DIR / "circuit/edges.h5",
            "--spatial-synapse-index-dir",
            EXTERNAL_DIR / "circuit/spatial_index_synapses",
            "--seed",
            0,
            "--population-name",
            "neuroglial",
            "--output-path",
            "neuroglial.connectivity.h5",
        ],
    )


def test_neuroglial_finalize():
    assert_cli_run(
        tested.attach_morphology_info_to_neuroglial_connectivity,
        [
            "--input-file-path",
            TMP_SONATA_DIR / "edges/neuroglial.connectivity.h5",
            "--output-file-path",
            "neuroglial.h5",
            "--astrocytes-path",
            FIN_SONATA_DIR / "nodes/glia.h5",
            "--microdomains-path",
            BUILD_DIR / "microdomains.h5",
            "--synaptic-data-path",
            EXTERNAL_DIR / "circuit/edges.h5",
            "--morph-dir",
            BUILD_DIR / "morphologies",
            "--seed",
            0,
        ],
    )


def test_glialglial_connectivity():
    assert_cli_run(
        tested.build_glialglial_connectivity,
        [
            "--astrocytes",
            FIN_SONATA_DIR / "nodes/glia.h5",
            "--touches-dir",
            BUILD_DIR / "connectome/touches",
            "--seed",
            0,
            "--population-name",
            "glialglial",
            "--output-connectivity",
            "glialglial.h5",
        ],
    )


def test_endfeet_meshes():
    assert_cli_run(
        tested.build_endfeet_surface_meshes,
        [
            "--config-path",
            BIONAME_DIR / "MANIFEST.yaml",
            "--vasculature-mesh-path",
            EXTERNAL_DIR / "atlas/vasculature.obj",
            "--gliovascular-connectivity-path",
            FIN_SONATA_DIR / "edges/gliovascular.h5",
            "--seed",
            0,
            "--output-path",
            "endfeet_meshes.h5",
        ],
    )


def test_synthesis():
    assert_cli_run(
        tested.synthesis,
        [
            "--config-path",
            BIONAME_DIR / "MANIFEST.yaml",
            "--tns-distributions-path",
            BIONAME_DIR / "tns_distributions.json",
            "--tns-parameters-path",
            BIONAME_DIR / "tns_parameters.json",
            "--tns-context-path",
            BIONAME_DIR / "tns_context.json",
            "--er-data-path",
            BIONAME_DIR / "er_data.json",
            "--astrocytes-path",
            FIN_SONATA_DIR / "nodes/glia.h5",
            "--microdomains-path",
            BUILD_DIR / "microdomains.h5",
            "--gliovascular-connectivity-path",
            FIN_SONATA_DIR / "edges/gliovascular.h5",
            "--neuroglial-connectivity-path",
            FIN_SONATA_DIR / "edges/neuroglial.h5",
            "--endfeet-meshes-path",
            BUILD_DIR / "endfeet_meshes.h5",
            "--neuronal-connectivity-path",
            EXTERNAL_DIR / "circuit/edges.h5",
            "--out-morph-dir",
            "morphologies",
            "--seed",
            0,
        ],
    )


def test_verify_circuit_integrity():
    assert_cli_run(
        main.verify_circuit_integrity,
        [
            BUILD_DIR / "ngv_config.json",
        ],
    )
