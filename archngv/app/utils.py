""" Miscellaneous utilities. """

import os
import yaml


def load_yaml(filepath):
    """ Load YAML file. """
    with open(filepath) as f:
        # TODO: verify config schema?
        return yaml.safe_load(f)


def ensure_dir(dirpath):
    """ Create folder if it is not there yet. """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
