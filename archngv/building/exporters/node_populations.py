"""Node populations io"""

import numpy as np
import pandas as pd
from voxcell import CellCollection

from archngv.core.constants import Population


def load_vasculature_node_population(filepath):
    """
    Args:
        filepath (str): SONATA NodePopulation filepath

    Returns:
        PointVasculature: The vasculature data structure
    """
    from vasculatureapi import PointVasculature

    cells = CellCollection.load_sonata(filepath)

    def get_prop(names):
        return cells.properties.loc[:, names].to_numpy()

    start_node_ids = get_prop("start_node_id")
    end_node_ids = get_prop("end_node_id")

    edge_properties = pd.DataFrame(
        {
            "start_node": start_node_ids,
            "end_node": end_node_ids,
            "type": get_prop("type"),
        },
        index=pd.MultiIndex.from_arrays(
            (get_prop("section_id"), get_prop("segment_id")),
            names=["section_id", "segment_id"],
        ),
    )
    uids = np.unique(np.hstack([start_node_ids, end_node_ids]))
    assert np.all(uids == np.arange(uids.size, dtype=uids.dtype))
    xyzd = np.empty((uids.size, 4), dtype=np.float32)
    xyzd[start_node_ids] = get_prop(["start_x", "start_y", "start_z", "start_diameter"])
    xyzd[end_node_ids] = get_prop(["end_x", "end_y", "end_z", "end_diameter"])
    node_properties = pd.DataFrame(data=xyzd, columns=["x", "y", "z", "diameter"])
    return PointVasculature(node_properties, edge_properties)


def save_vasculature_node_population(vasculature, filepath):
    """Exports the vasculature data structure into a SONATA NodePopulation where
    each node is an edge in the vasculature dataset. Point properties are stored
    on the edges as beg_property and end_property.

    Args:
        vasculature (Vasculature): input data structure
    """
    indices = vasculature.edge_properties.index
    points = vasculature.points
    diameters = vasculature.diameters
    start_node_ids, end_node_ids = vasculature.edges.T

    # for multiple columns assignment
    columns = ["start_x", "start_y", "start_z", "end_x", "end_y", "end_z"]
    properties = pd.DataFrame(index=np.arange(len(indices)), columns=columns)

    properties["type"] = vasculature.edge_types.astype(np.int32)
    properties["section_id"] = indices.get_level_values("section_id").to_numpy().astype(np.uint32)
    properties["segment_id"] = indices.get_level_values("segment_id").to_numpy().astype(np.uint32)

    properties["start_node_id"] = start_node_ids.astype(np.uint64)
    properties["end_node_id"] = end_node_ids.astype(np.uint64)

    properties[["start_x", "start_y", "start_z"]] = points[start_node_ids].astype(np.float32)
    properties[["end_x", "end_y", "end_z"]] = points[end_node_ids].astype(np.float32)
    properties["start_diameter"] = diameters[start_node_ids].astype(np.float32)
    properties["end_diameter"] = diameters[end_node_ids].astype(np.float32)

    cells = CellCollection(population_name=Population.VASCULATURE)
    cells.add_properties(properties)
    cells.save_sonata(filepath)


def export_astrocyte_population(filepath, cell_names, somata_positions, somata_radii, mtype):
    """Export cell data"""
    cell_names = np.asarray(cell_names, dtype=bytes)

    cells = CellCollection(population_name=Population.ASTROCYTES)
    cells.positions = somata_positions
    cells.properties["radius"] = somata_radii
    cells.properties["morphology"] = np.asarray(cell_names, dtype=str)
    cells.properties["mtype"] = mtype
    cells.properties["model_type"] = "biophysical"
    cells.save_sonata(filepath)
