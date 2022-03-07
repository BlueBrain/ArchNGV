"""Functions for conversion from old formats"""
from pathlib import Path

import h5py
import numpy as np

from archngv.app.utils import load_yaml, write_yaml


def merge_configuration_files(bioname_dir: Path, output_manifest_path: Path):
    """Merges the old configurations files into a single one.

    Args:
        bioname_dir: The path to the bioname dir with the config files
        output_manifest_path: The path to write the merged output manifest
    """

    configs_to_merge = {
        "cell_placement": "cell_placement.yaml",
        "microdomains": "microdomains.yaml",
        "gliovascular_connectivity": "gliovascular_connectivity.yaml",
        "endfeet_surface_meshes": "endfeet_area.yaml",
        "synthesis": "synthesis.yaml",
    }

    unified_config = load_yaml(bioname_dir / "MANIFEST.yaml")

    for config_name, config_file in configs_to_merge.items():
        unified_config[config_name] = load_yaml(bioname_dir / config_file)

    write_yaml(output_manifest_path, unified_config)


def convert_microdomains_to_generic_format(old_file_path: Path, new_file_path: Path):
    """Makes microdomain layout more generic, which allows adding more properties in the future.

    Args:
        old_file: Path to the old microdomains hdf5 file.
        new_file: Path to the output microdomains hdf5 file.
    """
    with h5py.File(old_file_path, mode="r") as old_file:
        with h5py.File(new_file_path, mode="w") as new_file:

            g_data = new_file.create_group("data", track_order=True)
            g_data.create_dataset("points", data=old_file["data"]["points"], dtype=np.float32)
            g_data.create_dataset(
                "triangle_data", data=old_file["data"]["triangle_data"], dtype=np.int64
            )
            g_data.create_dataset("neighbors", data=old_file["data"]["neighbors"], dtype=np.int64)

            g_offsets = new_file.create_group("offsets", track_order=True)
            g_offsets.create_dataset("points", data=old_file["offsets"][:, 0], dtype=np.int64)
            g_offsets.create_dataset(
                "triangle_data", data=old_file["offsets"][:, 1], dtype=np.int64
            )
            g_offsets.create_dataset("neighbors", data=old_file["offsets"][:, 2], dtype=np.int64)
