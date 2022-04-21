""" Miscellaneous utilities. """

import json
import os
from pathlib import Path
from typing import Any, Dict, Union

import click
import numpy
import yaml

REQUIRED_PATH = click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True)

PathLike = Union[str, Path]


def load_ngv_manifest(filepath: PathLike) -> Dict[str, Any]:
    """Loads a manifest configuration file.

    Args:
        filepath: The path to the manifest file

    Notes:

        There are two types of layouts that are supported for the ngv manifest:

        1. The standalone ngv circuit in which the config consists only of the ngv specific
            configuration.
        2. The combination of a regular circuit and an ngv circuit manifests. In this case, the
            entire ngv configuration is under the key 'ngv'.
    """
    manifest = load_yaml(filepath)
    return manifest["ngv"] if "ngv" in manifest else manifest


def load_yaml(filepath: PathLike) -> dict:
    """Load YAML file."""
    with open(filepath, mode="r", encoding="utf-8") as f:
        # TODO: verify config schema?
        return yaml.safe_load(f)


def write_yaml(filepath: Union[str, Path], data: dict) -> None:
    """Writes dict data to yaml"""

    class Dumper(yaml.SafeDumper):
        """Custom dumper that adds an empty line between root level entries"""

        def write_line_break(self, data=None):
            super().write_line_break(data)

            if len(self.indents) == 1:
                super().write_line_break()

    with open(filepath, mode="w", encoding="utf-8") as out_file:
        yaml.dump(data, out_file, Dumper=Dumper, sort_keys=False, default_flow_style=False)


def load_json(filepath: Union[str, Path]) -> dict:
    """Load json file"""
    with open(filepath, mode="r", encoding="utf-8") as f:
        return json.load(f)


def write_json(filepath: Union[str, Path], data: dict) -> None:
    """Write data to json file"""
    with open(filepath, mode="w", encoding="utf-8") as out_file:
        json.dump(data, out_file, indent=2)


def ensure_dir(dirpath):
    """Create folder if it is not there yet."""
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def choose_connectome(circuit):
    """Choose connectome from single-population SONATA circuit."""
    assert len(circuit.connectome) == 1
    return next(iter(circuit.connectome.values()))


def apply_parallel_function(function, data_generator):
    """Apply the function on the data generator in parallel and yield the results
    Notes:
        The results are unordered
    """
    import joblib

    return joblib.Parallel(verbose=150, n_jobs=-1)(
        joblib.delayed(function)(data) for data in data_generator
    )


def readonly_morphology(filepath, position):
    """It translates the morphology to its position inside the circuit
    space using the soma position.

    Args:
        filepath (str): Path to morphology file
        position (np.ndarray): Morphology offset

    Returns:
        readonly_morphology: morphio.Morphology
    """
    import morphio
    from morph_tool.transform import translate

    morphology = morphio.mut.Morphology(filepath)  # pylint: disable=no-member
    translate(morphology, position)
    morphology = morphology.as_immutable()

    return morphology


def random_generator(seed):
    """Returns random generator instance"""
    return numpy.random.default_rng(seed=seed)
