"""Grouped properties"""
from pathlib import Path
from typing import Dict

import h5py
import numpy as np


def export_grouped_properties(filepath: Path, properties: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Writes grouped properties into an hdf5 file.

    Args:
        filepath: Path to output file.
        properties:
            A dictionary the keys of which are property names and the values are dictionaries,
            containing two keys:
                - values: A numpy array with all the property data.
                - offsets: A numpy array of integers representing the offsets corresponding to the
                    groups in the values, or None if the dataset is linear without groups. If None,
                    the `values` will be added in `data` without a respective `offsets` dataset.

    Notes:
        The property values of the i-th group correspond to values[offsets[i]: offsets[i + 1]]
    """
    with h5py.File(filepath, mode="w") as f:

        g_data = f.create_group("data", track_order=True)
        g_offsets = f.create_group("offsets", track_order=True)

        for name, dct in properties.items():

            g_data.create_dataset(name, data=dct["values"])

            if dct["offsets"] is not None:
                g_offsets.create_dataset(name, data=dct["offsets"].astype(np.int64))
