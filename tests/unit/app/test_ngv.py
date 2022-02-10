from pathlib import Path
import tempfile
import numpy as np
from pathlib import Path
import pytest
import pandas as pd

import voxcell
import click.testing

import traceback
from archngv.app import ngv as tested


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
            "--bioname", BIONAME_DIR,
            "--output", "ngv_config.json",
        ]
    )


def test_assign_emodels():

    assert_cli_run(
        tested.assign_emodels,
        [
            "--input", TMP_SONATA_DIR / "nodes/glia.somata.h5",
            "--hoc", "template-filename",
            "--output", "output_nodes.h5",
        ]
    )


def test_cell_placement():

    # TODO: Create a click sonata file type so that if sth else is passed to throw
    # meaningful error.

    assert_cli_run(
        tested.cell_placement,
        [
            "--config", BIONAME_DIR / "cell_placement.yaml",
            "--atlas", EXTERNAL_DIR / "atlas",
            "--atlas-cache", ".atlas",
            "--vasculature", FIN_SONATA_DIR / "nodes/vasculature.h5",
            "--seed", 0,
            "--output", "output_nodes.h5",
        ]
    )


def test_finalize_astrocytes():

    assert_cli_run(
        tested.finalize_astrocytes,
        [
            "--somata-file", TMP_SONATA_DIR / "nodes/glia.somata.h5",
            "--emodels-file", TMP_SONATA_DIR / "nodes/glia.emodels.h5",
            "--output", "glia.h5",
        ]
    )


def test_microdomains():

    assert_cli_run(
        tested.build_microdomains,
        [
            "--config", BIONAME_DIR / "microdomains.yaml",
            "--astrocytes", FIN_SONATA_DIR / "nodes/glia.h5",
            "--atlas", EXTERNAL_DIR / "atlas",
            "--atlas-cache", ".atlas",
            "--seed", 0,
            "--output-dir", "microdomains_dir",
        ]
    )


def test_gliovascular_connectivity():

    assert_cli_run(
        tested.gliovascular_connectivity,
        [
            "--config", BIONAME_DIR / "gliovascular_connectivity.yaml",
            "--astrocytes", FIN_SONATA_DIR / "nodes/glia.h5",
            "--microdomains", BUILD_DIR / "microdomains/overlapping_microdomains.h5",
            "--vasculature", FIN_SONATA_DIR / "nodes/vasculature.h5",
            "--seed", 0,
            "--output", "gliovascular.h5",
        ]
    )


def test_gliovascular_finalize():

    assert_cli_run(
        tested.attach_endfeet_info_to_gliovascular_connectivity,
        [
            "--input-file", TMP_SONATA_DIR / "edges/gliovascular.connectivity.h5",
            "--output-file", "gliovascular.h5",
            "--astrocytes", FIN_SONATA_DIR / "nodes/glia.h5",
            "--endfeet-areas", BUILD_DIR / "endfeet_areas.h5",
            "--vasculature-sonata", FIN_SONATA_DIR / "nodes/vasculature.h5",
            "--morph-dir", BUILD_DIR / "morphologies",
        ]
    )


def test_neuroglial_connectivity():

    assert_cli_run(
        tested.build_neuroglial_connectivity,
        [
            "--neurons", EXTERNAL_DIR / "circuit/nodes.h5",
            "--astrocytes", FIN_SONATA_DIR / "nodes/glia.h5",
            "--microdomains", BUILD_DIR / "microdomains/overlapping_microdomains.h5",
            "--neuronal-connectivity", EXTERNAL_DIR / "circuit/edges.h5",
            "--seed", 0,
            "--output", "neuroglial.connectivity.h5",
        ]
    )


def test_neuroglial_finalize():

    assert_cli_run(
        tested.neuroglial_finalize,
        [
            "--input-file", TMP_SONATA_DIR / "edges/neuroglial.connectivity.h5",
            "--output-file", "neuroglial.h5",
            "--astrocytes", FIN_SONATA_DIR / "nodes/glia.h5",
            "--microdomains", BUILD_DIR / "microdomains/overlapping_microdomains.h5",
            "--synaptic-data", EXTERNAL_DIR / "circuit/edges.h5",
            "--morph-dir", BUILD_DIR / "morphologies",
            "--seed", 0
        ]
    )


def test_glialglial_connectivity():

    assert_cli_run(
        tested.build_glialglial_connectivity,
        [
            "--astrocytes", FIN_SONATA_DIR / "nodes/glia.h5",
            "--touches-dir", BUILD_DIR / "connectome/touches",
            "--seed", 0,
            "--output-connectivity", "glialglial.h5",
        ]
    )


def test_endfeet_areas():

    assert_cli_run(
        tested.build_endfeet_surface_meshes,
        [
            "--config-path", BIONAME_DIR / "endfeet_area.yaml",
            "--vasculature-mesh-path", EXTERNAL_DIR / "atlas/vasculature.obj",
            "--gliovascular-connectivity-path", FIN_SONATA_DIR / "edges/gliovascular.h5",
            "--seed", 0,
            "--output-path", "endfeet_areas.h5",
        ]
    )


def test_synthesis():

    assert_cli_run(
        tested.synthesis,
        [
            "--config-path", BIONAME_DIR / "synthesis.yaml",
            "--tns-distributions-path", BIONAME_DIR / "tns_distributions.json",
            "--tns-parameters-path", BIONAME_DIR / "tns_parameters.json",
            "--tns-context-path", BIONAME_DIR / "tns_context.json",
            "--er-data-path", BIONAME_DIR / "er_data.json",
            "--astrocytes-path", FIN_SONATA_DIR / "nodes/glia.h5",
            "--microdomains-path", BUILD_DIR / "microdomains/overlapping_microdomains.h5",
            "--gliovascular-connectivity-path", FIN_SONATA_DIR / "edges/gliovascular.h5",
            "--neuroglial-connectivity-path", FIN_SONATA_DIR / "edges/neuroglial.h5",
            "--endfeet-areas-path", BUILD_DIR / "endfeet_areas.h5",
            "--neuronal-connectivity-path", EXTERNAL_DIR / "circuit/edges.h5",
            "--out-morph-dir", "morphologies",
            "--seed", 0,
        ]
    )
