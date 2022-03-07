import tempfile

import h5py
import numpy as np
from numpy import testing as npt

from archngv.building.exporters import grouped_properties as tested
from archngv.core.datasets import GroupedProperties


def test_export_grouped_properties():

    properties = {
        "property1": {
            "values": np.array([0, 1, 2, 3], dtype=np.int32),
            "offsets": np.array([0, 2, 4], dtype=np.int64),
        },
        "property2": {
            "values": np.array(
                [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float32
            ),
            "offsets": np.array([0, 3, 5], dtype=np.int64),
        },
        "property3": {
            "values": np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
            "offsets": np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        },
    }

    with tempfile.NamedTemporaryFile(suffix=".h5") as tfile:

        filepath = tfile.name

        tested.export_grouped_properties(filepath, properties)

        with h5py.File(filepath, mode="r") as fp:

            for property_name, dct in properties.items():

                npt.assert_allclose(
                    fp["data"][property_name][:],
                    properties[property_name]["values"],
                )

                npt.assert_allclose(
                    fp["offsets"][property_name][:],
                    properties[property_name]["offsets"],
                )

        # test the core dataset that reads the grouped properties
        g = GroupedProperties(filepath)

        for property_name, dct in properties.items():

            expected_data = dct["values"]
            expected_offsets = dct["offsets"]

            assert g.get_property(property_name).dtype == expected_data.dtype

            n_groups = len(expected_offsets) - 1
            for i in range(n_groups):

                npt.assert_allclose(
                    g.get_property(property_name, i),
                    expected_data[expected_offsets[i] : expected_offsets[i + 1]],
                )
