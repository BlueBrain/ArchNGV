""" Microdomain expoerters functions """
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable

import numpy as np

from archngv.building.exporters.grouped_properties import export_grouped_properties

if TYPE_CHECKING:

    from archngv.core.datasets import Microdomain


def export_microdomains(
    filename: Path, domains: Iterable["Microdomain"], scaling_factors: np.ndarray
) -> None:
    """Export microdomain tessellation structure

    Args:
        filename: Path to output hdf5 file.
        domains: Microdomain iterable
        scaling_factors: The scaling factors that were used to scale the domains and make them
            overlapping.

    Notes:
        HDF5 Layout Hierarchy:
            data:
                points: array[float32, (N, 3)]
                    xyz coordinates of microdomain points
                triangle_data: array[int64, (M, 4)]
                    [polygon_id, v0, v1, v2]
                    The polygon the triangle belongs to and its vertices
                neighbors: array[int64, (K, 1)]
                    The neighbors to each triangle. Negative numbers signify a
                    bounding box wall.
                scaling_factors: array[float64, (G,)]

            offsets:
                Assuming there are G groups to be stored.
                points: array[int64, (G + 1,)]
                triangle_data: array[int64, (G + 1,)]
                neighbors: array[int64, (G + 1,)]

        The data of the i-th group in X dataset corresponds to:
            data[X][offsets[X][i]: offsets[X][i+1]]
    """
    n_domains = len(scaling_factors)

    properties_to_attributes = {
        "points": "points",
        "triangle_data": "triangle_data",
        "neighbors": "neighbor_ids",
    }

    property_dtypes = {"points": np.float32, "triangle_data": np.int64, "neighbors": np.int64}

    properties: Dict[str, Dict[str, Any]] = {
        name: {
            "values": [],
            "offsets": np.zeros((n_domains + 1), dtype=np.int64),
        }
        for name in properties_to_attributes
    }

    for index, dom in enumerate(domains):
        for property_name, attribute_name in properties_to_attributes.items():
            values = getattr(dom, attribute_name)
            attr = properties[property_name]
            attr["values"].append(values)
            attr["offsets"][index + 1] = attr["offsets"][index] + len(values)

    properties["points"]["values"] = np.vstack(properties["points"]["values"]).astype(
        property_dtypes["points"]
    )
    properties["triangle_data"]["values"] = np.vstack(properties["triangle_data"]["values"]).astype(
        property_dtypes["triangle_data"]
    )
    properties["neighbors"]["values"] = np.hstack(properties["neighbors"]["values"]).astype(
        property_dtypes["neighbors"]
    )

    properties["scaling_factors"] = {"values": scaling_factors.astype(np.float64), "offsets": None}

    export_grouped_properties(filename, properties)
