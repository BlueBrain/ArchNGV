""" Miscellaneous utilities. """

import json
import os
from pathlib import Path
from typing import Union

import click
import numpy
import yaml

REQUIRED_PATH = click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True)


def load_yaml(filepath):
    """Load YAML file."""
    with open(filepath, mode="r", encoding="utf-8") as f:
        # TODO: verify config schema?
        return yaml.safe_load(f)


def write_json(filepath: Union[str, Path], data: dict):
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
