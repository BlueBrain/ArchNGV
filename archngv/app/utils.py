import os
import yaml


def load_yaml(filepath):
    with open(filepath) as f:
        # TODO: verify config schema?
        return yaml.safe_load(f)


def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
