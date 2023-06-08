"""NGV config building functions"""
from json.decoder import JSONDecodeError
from pathlib import Path

import h5py
import libsonata
from bluepy_configfile.exceptions import BlueConfigError

from archngv.app.logger import LOGGER as L
from archngv.core.constants import Population
from archngv.core.sonata_readers import EdgesReader, NodesReader
from archngv.exceptions import NGVError


def build_ngv_config(root_dir: Path, manifest: dict) -> dict:
    """Returns an ngv sonata config including:
    Nodes:
        - Neurons
        - Glia
        - Vasculature

    Edges:
        - Neuronal
        - Gliovascular
        - Neuroglial
        - Glialglial
    """
    config = {
        "manifest": {
            "$CIRCUIT_DIR": "../",
            "$BASE_DIR": "$CIRCUIT_DIR/build",
        },
        "components": {},
        "networks": {"nodes": [], "edges": []},
    }

    common = manifest["common"]

    # add neuronal nodes and edges from existing circuit
    _add_neuronal_circuit(
        config=config,
        circuit_path=_make_abs(root_dir, common["base_circuit"]),
        neuron_config_filename=common["base_circuit_sonata"],
    )

    # the ngv specific nodes and edges of the current build
    _add_ngv_sonata_nodes_edges(config, root_dir, common)

    config["metadata"] = {"status": "partial"}

    return config


def _make_abs(parent_dir, *paths):
    path = Path(*paths)
    if not (str(path).startswith("$") or path.is_absolute()):
        return str(Path(parent_dir, path).resolve())
    return str(Path(*paths))


def _check_sonata_file(filepath, sonata_type):
    """Check if a h5 file is a sonata file."""
    if sonata_type not in ["nodes", "edges"]:
        raise NGVError(f"sonata_type must be 'nodes' or 'edges' not {sonata_type}")
    with h5py.File(filepath, "r") as h5:
        if sonata_type not in h5:
            raise NGVError(f"{filepath} is not a sonata file")
    return filepath


def _find_neuron_config(circuit_path, neuron_config_filename):
    """Returns the absolute path to the neuronal circuit config depending on what
    type of config it is (BlueConfig, Sonata config etc.).
    """
    if neuron_config_filename is not None:
        config_filepath = Path(circuit_path, neuron_config_filename)
        if not config_filepath.exists():
            raise NGVError(f"Neuron circuit config {config_filepath.resolve()} does not exist")

    else:
        default_names = ["CircuitConfig", "BlueConfig", "circuit_config.json"]
        for default_name in default_names:
            config_filepath = Path(circuit_path) / default_name
            if config_filepath.exists():
                break
        else:
            raise NGVError(f"Neuron circuit config not found in {config_filepath}")

    return config_filepath


def _add_neuronal_circuit(config, circuit_path, neuron_config_filename):
    config_filepath = _find_neuron_config(circuit_path, neuron_config_filename)
    L.warning("Use %s as neuronal config file", config_filepath)
    try:
        # must be a sonata file i.e.: absolute path.
        from bluepy_configfile.configfile import BlueConfig

        with open(config_filepath) as f:
            blue_config = BlueConfig(f)

        config["networks"]["nodes"].extend(
            [
                {
                    "nodes_file": _check_sonata_file(blue_config.Run.CircuitPath, "nodes"),
                }
            ]
        )

        config["networks"]["edges"].extend(
            [
                {
                    "edges_file": _check_sonata_file(blue_config.Run.nrnPath, "edges"),
                }
            ]
        )

        morph_type = blue_config.Run.get("MorphologyType", "neurolucida-asc").lower()
        morph_path = blue_config.Run.MorphologyPath

        if morph_type == "neurolucida-asc":
            morph_path = str(Path(morph_path, "ascii"))

        if morph_type == "swc":
            config["components"] = {"morphologies_dir": morph_path}
        else:
            config["components"] = {"alternate_morphologies": {morph_type: morph_path}}

    except BlueConfigError as bc_e:
        try:
            from bluepysnap import Config

            tmp_config = Config(config_filepath, libsonata.CircuitConfig).to_dict()

            if len(tmp_config["networks"]["nodes"]) > 1:
                raise NGVError(
                    "Only neuron circuits with a single node population are allowed."
                ) from bc_e

            if len(tmp_config["networks"]["edges"]) > 1:
                raise NGVError(
                    "Only neuron circuits with a single edge population are allowed."
                ) from bc_e

            tmp_config.pop("manifest", None)
            config.update(tmp_config)
        except (JSONDecodeError, KeyError) as e:
            raise NGVError(f"{config_filepath} is not a bbp/sonata circuit config file") from e

    neuronal_nodes = config["networks"]["nodes"][0]
    neuron_node_population = NodesReader(neuronal_nodes["nodes_file"]).name

    if "populations" not in neuronal_nodes:
        neuronal_nodes["populations"] = {}

    neuronal_nodes["populations"][neuron_node_population] = {
        "type": "biophysical",
    }

    # move the global components inside the neuronal node population
    # this is only valid for single population files
    if "components" in config:
        neuronal_nodes["populations"][neuron_node_population].update(config["components"])
        config.pop("components", None)

    neuronal_edges = config["networks"]["edges"][0]
    neuron_edge_population = EdgesReader(neuronal_edges["edges_file"]).name

    if "populations" not in neuronal_edges:
        neuronal_edges["populations"] = {}

    neuronal_edges["populations"][neuron_edge_population] = {"type": Population.NEURONAL}

    return config


def _add_ngv_sonata_nodes_edges(config: dict, root_dir: Path, manifest: dict) -> None:
    """
    Add the ngv nodes and connectivities. They need to be predefined instead of
    searched because the ngv config should be able to be created at any time for
    accessing data from partial circuits that a subset of rules are ran.
    """
    # fmt: off
    config["networks"]["nodes"].extend(
        [
            {
                "nodes_file": "$BASE_DIR/sonata/networks/nodes/vasculature/nodes.h5",
                "populations": {
                    "vasculature": {
                        "type": Population.VASCULATURE,
                        "vasculature_file": _make_abs(root_dir, manifest["vasculature"]),
                        "vasculature_mesh": _make_abs(root_dir, manifest["vasculature_mesh"]),
                    }
                },
            },
            {
                "nodes_file": "$BASE_DIR/sonata/networks/nodes/astrocytes/nodes.h5",
                "populations": {
                    "astrocytes": {
                        "type": Population.ASTROCYTES,
                        "alternate_morphologies": {"h5v1": "$BASE_DIR/morphologies"},
                        "microdomains_file": "$BASE_DIR/microdomains.h5",
                    }
                },
            },
        ]
    )
    # fmt: on

    config["networks"]["edges"].extend(
        [
            {
                "edges_file": "$BASE_DIR/sonata/networks/edges/neuroglial/edges.h5",
                "populations": {"neuroglial": {"type": Population.NEUROGLIAL}},
            },
            {
                "edges_file": "$BASE_DIR/sonata/networks/edges/glialglial/edges.h5",
                "populations": {"glialglial": {"type": Population.GLIALGLIAL}},
            },
            {
                "edges_file": "$BASE_DIR/sonata/networks/edges/gliovascular/edges.h5",
                "populations": {
                    "gliovascular": {
                        "type": Population.GLIOVASCULAR,
                        "endfeet_meshes_file": "$BASE_DIR/endfeet_meshes.h5",
                    }
                },
            },
        ]
    )
