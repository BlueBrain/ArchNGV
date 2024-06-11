# SPDX-License-Identifier: Apache-2.0

"""Functions for conversion from old formats"""
import shutil
from pathlib import Path

import h5py
import morphio
import numpy as np

from archngv.app.utils import load_yaml, write_yaml
from archngv.building.exporters import add_properties_to_edge_population, export_grouped_properties
from archngv.core import datasets


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


def merge_microdomain_files(microdomains_dir: Path, output_file_path: Path):
    """Merges regular and overlapping microdomain files into a single overlapping microdomain file,
    which also contains the scaling factors to reconstruct the regular tessellation if needed.

    Args:
        microdomains_dir: Path to the directory containing the two domain files
        output_file_path: Path to output microdomains hdf5 file
    """

    def get_data_slice(data, offsets, index):
        return data[offsets[index] : offsets[index + 1]]

    regular_microdomains_file_path = Path(microdomains_dir, "microdomains.h5")
    overlapping_microdomains_file_path = Path(microdomains_dir, "overlapping_microdomains.h5")

    with h5py.File(regular_microdomains_file_path, "r") as r_domains:
        with h5py.File(overlapping_microdomains_file_path, "r") as o_domains:
            with h5py.File(output_file_path, "w") as out_file:
                num_domains = len(r_domains["offsets"]["points"]) - 1

                assert (
                    len(o_domains["offsets"]["points"]) - 1 == num_domains
                ), "Different number of domains in the two files."

                # regular
                r_data = r_domains["data"]
                r_offs = r_domains["offsets"]

                # overlapping
                o_data = o_domains["data"]
                o_offs = o_domains["offsets"]

                g_data = out_file.create_group("data", track_order=True)
                g_offs = out_file.create_group("offsets", track_order=True)

                for name in ("points", "triangle_data", "neighbors"):
                    g_data.create_dataset(name, data=o_data[name])
                    g_offs.create_dataset(name, data=o_offs[name])

                # triangle_data and neighbors should be identical between the two tessellations
                for name in ("triangle_data", "neighbors"):
                    assert np.allclose(
                        r_data[name][:], o_data[name][:]
                    ), f"{name} data entry is not identical between the two tessellations."
                    assert np.allclose(
                        r_offs[name][:], o_offs[name][:]
                    ), f"{name} offsets entry is not identical between the two tessellations."

                # we can reverse calculate the scaling factors from the points of the two datasets
                # scaled_points = scaling_factor * (points - centroid) + centroid
                scaling_factors = np.empty(num_domains, dtype=float)

                for domain_index in range(num_domains):
                    r_points = get_data_slice(r_data["points"], r_offs["points"], domain_index)
                    o_points = get_data_slice(o_data["points"], o_offs["points"], domain_index)

                    centroid = np.mean(r_points, axis=0)

                    o_points_origin = o_points - centroid
                    r_points_origin = r_points - centroid

                    not_zero = ~np.isclose(r_points_origin, 0.0) | ~np.isclose(o_points_origin, 0.0)

                    assert not_zero.any(), (
                        f"Regular points    : {r_points_origin}\n"
                        f"Overlapping points: {o_points_origin}\n"
                        f"Regular centroid  : {np.mean(r_points, axis=0)}\n"
                        f"Overlap centroid  : {np.mean(o_points, axis=0)}\n"
                    )

                    scaling_factors[domain_index] = np.mean(
                        o_points_origin[not_zero] / r_points_origin[not_zero]
                    )

                g_data.create_dataset("scaling_factors", data=scaling_factors)


def convert_endfeet_to_generic_format(old_file_path: Path, new_file_path: Path) -> None:
    """Convert endfeet surface meshes file to a grouped properties one"""
    properties = {}

    with h5py.File(old_file_path, "r") as o_endfeet:
        n_endfeet = len(o_endfeet["attributes"]["surface_area"])

        for name in ("surface_area", "surface_thickness", "unreduced_surface_area"):
            dset = o_endfeet["attributes"][name]

            assert (
                len(dset) == n_endfeet
            ), f"Mismatching dataset {name} size: Expected {n_endfeet} but got {len(dset)}."

            properties[name] = {
                "values": dset[:],
                "offsets": None,
            }

        for name in ("points", "triangles"):
            properties[name] = {
                "values": [],
                "offsets": np.zeros(n_endfeet + 1, dtype=np.int64),
            }

        for index in range(n_endfeet):
            g = o_endfeet["objects"][f"endfoot_{index}"]

            for name in ("points", "triangles"):
                values = g[name][:]
                properties[name]["values"].append(values)
                properties[name]["offsets"][index + 1] = properties[name]["offsets"][index] + len(
                    values
                )

        properties["points"]["values"] = np.vstack(properties["points"]["values"]).astype(
            np.float32
        )
        properties["triangles"]["values"] = np.vstack(properties["triangles"]["values"]).astype(
            np.int64
        )

        export_grouped_properties(new_file_path, properties)


def add_astrocyte_segment_center_property(
    astrocytes_file_path: Path,
    neuroglial_file_path: Path,
    morphologies_dir: Path,
    output_file_path: Path,
) -> None:
    """Adds the properties astrocyte_center_[x|y|z] into the neuroglial edge population.

    Args:
        astrocytes_file_path: Path to astrocytes SONATA node population.
        neuroglial_file_path: Path to neuroglial SONATA edge population.
        morphologies_dir: Path to morphologies directory.
        output_file_path: Path to output file.
    """
    astrocytes = datasets.CellData(astrocytes_file_path)
    neuroglial = datasets.NeuroglialConnectivity(neuroglial_file_path)

    astrocyte_positions = astrocytes.positions()
    astrocyte_names = astrocytes.get_property("morphology")
    astrocyte_section_ids = neuroglial.get_property("astrocyte_section_id")
    astrocyte_segment_ids = neuroglial.get_property("astrocyte_segment_id")
    astrocyte_segment_offsets = neuroglial.get_property("astrocyte_segment_offset")

    astrocyte_segment_center_points = np.zeros((len(neuroglial), 3), dtype=np.float32)

    for astrocyte_id in range(len(astrocytes)):
        edge_ids = neuroglial.efferent_edges(astrocyte_id)

        if len(edge_ids) == 0:
            continue

        morph = morphio.Morphology(Path(morphologies_dir, astrocyte_names[astrocyte_id] + ".h5"))

        points = morph.points

        section_offsets = morph.section_offsets

        section_ids = astrocyte_section_ids[edge_ids]
        section_begs = section_offsets[section_ids]

        segment_ids = astrocyte_segment_ids[edge_ids]
        segment_beg_points = points[section_begs + segment_ids]
        segment_end_points = points[section_begs + segment_ids + 1]

        axes = segment_end_points - segment_beg_points
        axes /= np.linalg.norm(axes, axis=1)[:, np.newaxis]

        # expand one dimension to broadcast with the N x 3 point arrays
        segment_offsets = astrocyte_segment_offsets[edge_ids, np.newaxis]

        astrocyte_segment_center_points[edge_ids] = (
            segment_beg_points + segment_offsets * axes + astrocyte_positions[astrocyte_id]
        )

    shutil.copyfile(neuroglial_file_path, output_file_path)

    add_properties_to_edge_population(
        output_file_path,
        neuroglial.name,
        {
            "astrocyte_center_x": astrocyte_segment_center_points[:, 0],
            "astrocyte_center_y": astrocyte_segment_center_points[:, 1],
            "astrocyte_center_z": astrocyte_segment_center_points[:, 2],
        },
    )
