""" Miscellaneous utilities. """

import os
import click
import yaml


REQUIRED_PATH = click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True)


def load_yaml(filepath):
    """ Load YAML file. """
    with open(filepath) as f:
        # TODO: verify config schema?
        return yaml.safe_load(f)


def ensure_dir(dirpath):
    """ Create folder if it is not there yet. """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def choose_connectome(circuit):
    """ Choose connectome from single-population SONATA circuit. """
    assert len(circuit.connectome) == 1
    return next(iter(circuit.connectome.values()))
