"""Grouped properties"""
from pathlib import Path
from typing import Dict

import h5py
import numpy as np


def export_grouped_properties(filepath: Path, properties: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Writes groupes properties into an hdf5 file.

    Args:
        filepath: Path to output file.
        properties:
            A dictionary the keys of which are property names and the values are dictionaries,
            containing two keys:
                - values: A numpy array with all the property data.
                - offsets: An integer numpy array with the offsets corresponding to the groups in
                    the values.

    Note:
        The property values of the i-th group correspond to values[offsets[i]: offsets[i + 1]]
    """
    with h5py.File(filepath, mode="w") as f:

        g_data = f.create_group("data", track_order=True)
        g_offsets = f.create_group("offsets", track_order=True)

        for name, dct in properties.items():

            g_data.create_dataset(name, data=dct["values"])
            g_offsets.create_dataset(name, data=dct["offsets"].astype(np.int64))
