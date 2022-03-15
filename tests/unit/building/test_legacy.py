import tempfile
from pathlib import Path

import h5py
import numpy as np
from numpy import testing as npt

from archngv.app.utils import load_yaml
from archngv.building import legacy as tested


DATA_DIR = Path(__file__).parent.resolve() / "data"


def test_merge_configuration_files():

    expected_manifest = {
        "common": {
            "log_level": "WARNING",
            "seed": 0,
            "atlas": "atlas-path",
            "vasculature": "vasculature-path",
            "vasculature_mesh": "vasculature-mesh-path",
            "base_circuit": "base-circuit-path",
            "base_circuit_sonata": "base-sonata-path",
            "base_circuit_cells": "base-circuit-nodes",
            "base_circuit_connectome": "base-circuit-edges",
        },
        "assign_emodels": {"templates_dir": "emodels-path", "hoc_template": "astrocyte"},
        "cell_placement": {
            "density": "[density]astrocytes",
            "soma_radius": [5.6, 0.74, 0.1, 20],
            "Energy": {"potentials": {"spring": [32.0, 1.0]}},
            "MetropolisHastings": {
                "n_initial": 10,
                "beta": 0.01,
                "ntrials": 3,
                "cutoff_radius": 60.0,
            },
        },
        "microdomains": {"overlap_distribution": {"type": "normal", "values": [0.1, 0.01]}},
        "gliovascular_connectivity": {
            "graph_targeting": {"linear_density": 0.17},
            "connection": {
                "reachout_strategy": "maximum_reachout",
                "endfeet_distribution": [2, 2, 1, 5],
            },
            "surface_targeting": {},
        },
        "endfeet_surface_meshes": {
            "fmm_cutoff_radius": 1000.0,
            "area_distribution": [192.0, 160.0, 0.0, 1000.0],
            "thickness_distribution": [0.97, 0.1, 0.01, 2.0],
        },
        "synthesis": {
            "perimeter_distribution": {
                "enabled": True,
                "statistical_model": {"slope": 2.0, "intercept": 1.0, "standard_deviation": 1.0},
                "smoothing": {"window": [1.0, 1.0, 1.0, 1.0, 1.0]},
            }
        },
    }

    with tempfile.NamedTemporaryFile(suffix=".yaml") as tfile:

        out_merged_filepath = tfile.name

        tested.merge_configuration_files(
            bioname_dir=DATA_DIR / "legacy/merge_configuration_files/old_bioname",
            output_manifest_path=out_merged_filepath,
        )

        actual_manifest = load_yaml(out_merged_filepath)
        assert expected_manifest == actual_manifest, (actual_manifest, expected_manifest)


def test_convert_microdomains_to_generic_format():

    old_file_path = DATA_DIR / "legacy/convert_microdomains_to_generic_format/microdomains.h5"

    with tempfile.NamedTemporaryFile(suffix=".h5") as tfile:

        new_file_path = tfile.name
        tested.convert_microdomains_to_generic_format(
            old_file_path=old_file_path, new_file_path=new_file_path
        )

        with h5py.File(old_file_path, "r") as old_file:
            with h5py.File(new_file_path, "r") as new_file:

                npt.assert_allclose(old_file["data"]["points"][:], new_file["data"]["points"][:])
                npt.assert_allclose(
                    old_file["data"]["triangle_data"][:], new_file["data"]["triangle_data"][:]
                )
                npt.assert_allclose(
                    old_file["data"]["neighbors"][:], new_file["data"]["neighbors"][:]
                )

                npt.assert_allclose(
                    old_file["offsets"][:],
                    np.column_stack(
                        (
                            new_file["offsets"]["points"],
                            new_file["offsets"]["triangle_data"],
                            new_file["offsets"]["neighbors"],
                        )
                    ),
                )


def test_merge_microdomains():
    def get_slice(h5_domains, dataset_name, i):

        data = h5_domains["data"][dataset_name]
        offs = h5_domains["offsets"][dataset_name]

        return data[offs[i] : offs[i + 1]]

    microdomains_dir = DATA_DIR / "legacy/merge_microdomain_files"

    path1 = microdomains_dir / "microdomains.h5"
    path2 = microdomains_dir / "overlapping_microdomains.h5"

    with tempfile.NamedTemporaryFile(suffix=".h5") as tfile:

        output_path = tfile.name

        tested.merge_microdomain_files(microdomains_dir, output_path)

        with h5py.File(output_path, "r") as merged_domains:
            with h5py.File(path1, "r") as r_domains:
                with h5py.File(path2, "r") as o_domains:

                    scaling_factors = merged_domains["data"]["scaling_factors"][:]

                    for i, scaling_factor in enumerate(scaling_factors):

                        r_points = get_slice(r_domains, "points", i)
                        o_points = get_slice(o_domains, "points", i)

                        centroid = np.mean(o_points, axis=0)
                        reconstructed_points = (o_points - centroid) / scaling_factor + centroid

                        npt.assert_allclose(reconstructed_points, r_points, atol=1e-5)

                        npt.assert_allclose(
                            get_slice(merged_domains, "triangle_data", i),
                            get_slice(r_domains, "triangle_data", i),
                        )
                        npt.assert_allclose(
                            get_slice(merged_domains, "triangle_data", i),
                            get_slice(o_domains, "triangle_data", i),
                        )
                        npt.assert_allclose(
                            get_slice(merged_domains, "neighbors", i),
                            get_slice(r_domains, "neighbors", i),
                        )
                        npt.assert_allclose(
                            get_slice(merged_domains, "neighbors", i),
                            get_slice(o_domains, "neighbors", i),
                        )
