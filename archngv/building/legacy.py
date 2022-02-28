"""Functions for conversion from old formats"""
from pathlib import Path

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
