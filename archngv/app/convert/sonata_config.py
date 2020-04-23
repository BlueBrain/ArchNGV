'''
Export input / output paths from NGVConfig to SONATA config
'''
from pathlib import Path
from json.decoder import JSONDecodeError

import click
import h5py

from bluepy_configfile.exceptions import BlueConfigError

from archngv.exceptions import NGVError
from archngv.app.utils import REQUIRED_PATH
from archngv.app.logger import LOGGER as L


def _check_sonata_file(filepath, sonata_type):
    """Check if a h5 file is a sonata file."""
    if sonata_type not in ["nodes", "edges"]:
        raise NGVError("sonata_type must be 'nodes' or 'edges' not %s" % sonata_type)
    with h5py.File(filepath, "r") as h5:
        if sonata_type not in h5:
            raise NGVError("%s is not a sonata file" % filepath)
    return filepath


def _find_neuron_config(circuit_path, neuron_config_filename):
    if neuron_config_filename is not None:
        config_filepath = Path(circuit_path) / neuron_config_filename
        if not config_filepath.exists():
            raise NGVError("Neuron circuit config {} does not exists".format(config_filepath))
    else:
        default_names = ["CircuitConfig", "circuit_config.json"]
        for default_name in default_names:
            config_filepath = Path(circuit_path) / default_name
            if config_filepath.exists():
                break
        else:
            raise NGVError("Neuron circuit config not found in {} ".format(circuit_path))
    return config_filepath


def _find_neuron_files(circuit_path, neuron_config_filename):
    config_filepath = _find_neuron_config(circuit_path, neuron_config_filename)
    L.warning("Use %s as neuronal config file", config_filepath)

    try:
        # must be a sonata file i.e.: absolute path.
        from bluepy_configfile.configfile import BlueConfig
        config = BlueConfig(open(config_filepath))
        nodes = [_check_sonata_file(config.Run.CircuitPath, "nodes")]
        edges = [_check_sonata_file(config.Run.nrnPath, "edges")]
        morph = config.Run.MorphologyPath
        return edges, nodes, morph
    except BlueConfigError:
        try:
            from bluepysnap import Config
            config = Config(config_filepath).resolve()
            nodes = [node["nodes_file"] for node in config["networks"]["nodes"]]
            edges = [edge["edges_file"] for edge in config["networks"]["edges"]]
            morph = config["components"]["morphologies_dir"]
            return edges, nodes, morph
        except (JSONDecodeError, KeyError):
            raise NGVError("{} is not a bbp/sonata circuit config file".format(config_filepath))


@click.command(help=__doc__)
@click.argument('ngv-config', type=REQUIRED_PATH)
@click.option('-n', '--neuron-config-filename', type=str, default=None, show_default=True)
@click.option('-o', '--output-file', type=str)
def cmd(ngv_config, neuron_config_filename, output_file):
    """Convert a ngv config to a sonata extended config_file"""
    import json
    from archngv.core.config import NGVConfig

    ngv_config = NGVConfig.from_file(ngv_config)
    neuron_config = ngv_config.input_paths("microcircuit_path")
    parent = Path(ngv_config.parent_directory) / ngv_config.experiment_name
    edges, nodes, morph = _find_neuron_files(neuron_config, neuron_config_filename)

    for file in Path(parent, "sonata/nodes").iterdir():
        nodes.append("$NETWORK_DIR/sonata/nodes/" + file.name)
    for file in Path(parent, "sonata/edges").iterdir():
        edges.append("$NETWORK_DIR/sonata/edges/" + file.name)

    def _create_node(node_file):
        return {"nodes_file": node_file, "node_types_file": None}

    def _create_edge(edge_file):
        return {"edges_file": edge_file, "edge_types_file": None}

    sonata_config = {
        "manifest": {
            "$BASE_DIR": str(parent),
            "$COMPONENT_DIR": "$BASE_DIR",
            "$NETWORK_DIR": "$BASE_DIR"
        },
        "components": {
            "morphologies_dir": morph
        },
        "networks": {
            "nodes": [_create_node(node) for node in nodes],
            "edges": [_create_edge(edge) for edge in edges],
            "vasculature": {
                "vasculature_file": ngv_config.input_paths("vasculature"),
                "vasculature_mesh_file": ngv_config.input_paths("vasculature_mesh")
            },
            "astrocytes": [
                {
                    "population": "astrocytes",
                    "microdomains_file": ngv_config.output_paths("microdomains"),
                    "microdomains_overlapping_file": ngv_config.output_paths(
                        "overlapping_microdomains"),
                    "endfeet_file": ngv_config.output_paths("endfeet_areas"),
                    "endfeet_data_file": ngv_config.output_paths("gliovascular_data"),
                    "morphologies_dir": ngv_config.output_paths("morphology"),
                }
            ],
            "gliovascular": ngv_config.output_paths("gliovascular_connectivity"),
            "atlases": {
                    "intensity": ngv_config.input_paths("voxelized_intensity"),
                    "brain_regions": ngv_config.input_paths("voxelized_brain_regions"),
            }
        }
    }
    with open(output_file, 'w') as f:
        json.dump(sonata_config, f, indent=2)
