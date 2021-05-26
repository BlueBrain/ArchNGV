"""Export input / output paths to a sonata extended config file : NGVConfig"""
from pathlib import Path
from json.decoder import JSONDecodeError

import click
import h5py
from bluepy_configfile.exceptions import BlueConfigError
from archngv.exceptions import NGVError
from archngv.app.logger import LOGGER as L
from archngv.core.constants import Population
from archngv.core.sonata_readers import NodesReader, EdgesReader


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
        except (JSONDecodeError, KeyError) as e:
            raise NGVError(
                "{} is not a bbp/sonata circuit config file".format(config_filepath)) from e


@click.command(help=__doc__)
@click.option('--bioname', help='Path to bioname folder', required=True)
@click.option('-o', '--output', help='Path to output file (JSON)', required=True)
def cmd(bioname, output):
    # pylint: disable=missing-docstring
    import json
    import os.path

    from archngv.app.utils import load_yaml

    def _load_config(name):
        return load_yaml(os.path.join(bioname, '%s.yaml' % name))

    def _make_abs(parent_dir, *paths):
        path = Path(*paths)
        if not (str(path).startswith("$") or path.is_absolute()):
            return str(Path(parent_dir, path).resolve())
        return str(Path(*paths))

    manifest = _load_config('MANIFEST')['common']
    bioname = Path(bioname).resolve()

    cell_placement_config = _load_config('cell_placement')
    synthesis_config = _load_config('synthesis')
    synthesis_config['endfeet_area_reconstruction'] = _load_config('endfeet_area')

    # neuronal nodes, connectivities and morphologies_dir path
    edges, nodes, morph = _find_neuron_files(
        manifest['base_circuit'], manifest['base_circuit_sonata'])

    # TODO : if more_itertools becomes a deps use more_itertools.one here instead
    if len(nodes) == 1:
        neuron_node_population = NodesReader(nodes[0]).name
    else:
        raise NGVError("Only neuron circuit with a single node population are allowed.")

    if len(edges) == 1:
        neuron_edge_population = EdgesReader(edges[0]).name
    else:
        raise NGVError("Only neuron circuit with a single edge population are allowed.")

    def _create_node(node_file):
        return {"nodes_file": node_file, "node_types_file": None}

    def _create_edge(edge_file):
        return {"edges_file": edge_file, "edge_types_file": None}

    # add the ngv nodes and connectivities
    # they need to be predefined instead of searched because the ngv config
    # should be able to be created at any time for accessing data from partial
    # circuits that a subset of rules are ran.
    nodes.extend([f'$NETWORK_DIR/sonata/nodes/{node}.h5' for node in ['glia', 'vasculature']])
    edges.extend([f'$NETWORK_DIR/sonata/edges/{edge}.h5' for edge in ['neuroglial', 'glialglial', 'gliovascular']])

    config = {
        "manifest": {
            "$CIRCUIT_DIR": "../",
            "$BUILD_DIR": "$CIRCUIT_DIR/build",
            "$COMPONENT_DIR": "$BUILD_DIR",
            "$NETWORK_DIR": "$BUILD_DIR"
        },
        "circuit_dir": "$CIRCUIT_DIR",
        "components": {
            "morphologies_dir": morph
        },
        "networks": {
            "nodes": [_create_node(node) for node in nodes],
            "edges": [_create_edge(edge) for edge in edges]
        },
        "cells": {
            "protoplasmic_astrocytes": {
                "population": Population.ASTROCYTES,
                "microdomains_file": "$BUILD_DIR/microdomains/microdomains.h5",
                "microdomains_overlapping_file": "$BUILD_DIR/microdomains/overlapping_microdomains.h5",
                "morphologies_dir": "$BUILD_DIR/morphologies"
            },
            Population.NEURONS: {"population": neuron_node_population},
            Population.VASCULATURE: {
                "population": Population.VASCULATURE,
                "vasculature_file": _make_abs(bioname, manifest["vasculature"]),
                "vasculature_mesh_file": _make_abs(bioname, manifest["vasculature_mesh"])
            }
        },
        "connectivities": {
            Population.NEURONAL: {"population": neuron_edge_population},
            Population.GLIOVASCULAR: {"population": Population.GLIOVASCULAR,
                                      "endfeet_areas": "$BUILD_DIR/endfeet_areas.h5"
                                      },
            Population.NEUROGLIAL: {"population": Population.NEUROGLIAL},
            Population.GLIALGLIAL: {"population": Population.GLIALGLIAL}
        },
        "atlases": {
            "intensity": _make_abs(bioname, manifest['atlas'],
                                   f"{cell_placement_config['density']}.nrrd"),
            "brain_regions": _make_abs(bioname, manifest['atlas'], 'brain_regions.nrrd'),
        },
        'parameters': {
            'cell_placement': _load_config('cell_placement'),
            'microdomain_tesselation': _load_config('microdomains'),
            'gliovascular_connectivity': _load_config('gliovascular_connectivity'),
            'neuroglial_connectivity': {},
            'synthesis': synthesis_config,
        }
    }

    with open(output, 'w') as f:
        json.dump(config, f, indent=2)
