import tempfile

import h5py
import numpy as np
import pytest
from numpy import testing as npt

from archngv.building import exporters as tested
from archngv.core.datasets import GroupedProperties


@pytest.fixture(scope="session")
def properties():
    return {
        "property1": {
            "values": np.array([0, 1, 2, 3], dtype=np.int32),
            "offsets": np.array([0, 2, 4], dtype=np.int64),
        },
        "property2": {
            "values": np.array([0, 1, 2, 3], dtype=np.uint32),
            "offsets": np.array([0, 0, 1, 2, 2, 3, 3, 4], dtype=np.int64),
        },
        "property3": {
            "values": np.array(
                [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float32
            ),
            "offsets": np.array([0, 3, 5], dtype=np.int64),
        },
        "property4": {
            "values": np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
            "offsets": np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        },
        "property5": {
            "values": np.array([10.0, 9.0, 8.0, 7.0, 6.0], dtype=float),
            "offsets": None,
        },
        "property6": {
            "values": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
            "offsets": None,
        },
    }


@pytest.fixture(scope="session")
def output_file(properties, tmpdir_factory):
    filepath = tmpdir_factory.mktemp(__name__).join("output.h5")
    tested.export_grouped_properties(filepath, properties)
    return filepath


def test_grouped_properties__hdf5_file(properties, output_file):
    with h5py.File(output_file, mode="r") as fp:
        for property_name, dct in properties.items():
            npt.assert_allclose(
                fp["data"][property_name][:],
                properties[property_name]["values"],
            )

            if dct["offsets"] is None:
                assert property_name not in fp["offsets"]
            else:
                npt.assert_allclose(
                    fp["offsets"][property_name][:],
                    properties[property_name]["offsets"],
                )


def test_export_grouped_properties_api(properties, output_file):
    g = GroupedProperties(output_file)

    for property_name, dct in properties.items():
        expected_data, expected_offsets = dct["values"], dct["offsets"]

        assert g.get(property_name).dtype == expected_data.dtype

        n_groups = len(expected_data) if expected_offsets is None else len(expected_offsets) - 1

        for i in range(n_groups):
            values = g.get(property_name, i)

            expected_values = (
                expected_data[i]
                if expected_offsets is None
                else expected_data[expected_offsets[i] : expected_offsets[i + 1]]
            )

            npt.assert_allclose(values, expected_values)
