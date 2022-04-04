""" Endfeetome exporters """
import logging
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np

from archngv.building.exporters.grouped_properties import export_grouped_properties
from archngv.core.datasets import EndfootMesh
from archngv.exceptions import NGVError

L = logging.getLogger(__name__)


def export_endfeet_meshes(filename: Path, endfeet: Iterator[EndfootMesh], n_endfeet: int) -> None:
    """Export endfeet meshes as grouped properties

    Args:
        filename: Output file path.
        endfeet: Iterable of EndfootMesh instances.
        n_endfeet: The size of the endfeet iterable.
    """

    property_dtypes = {
        "points": np.float32,
        "triangles": np.int64,
        "surface_area": np.float32,
        "unreduced_surface_area": np.float32,
        "surface_thickness": np.float32,
    }

    properties: Dict[str, Dict[str, Any]] = {
        "points": {
            "values": [[] for _ in range(n_endfeet)],
            "offsets": np.zeros(n_endfeet + 1, dtype=np.int64),
        },
        "triangles": {
            "values": [[] for _ in range(n_endfeet)],
            "offsets": np.zeros(n_endfeet + 1, dtype=np.int64),
        },
        "surface_area": {
            "values": np.zeros(n_endfeet, dtype=property_dtypes["surface_area"]),
            "offsets": None,
        },
        "unreduced_surface_area": {
            "values": np.zeros(n_endfeet, dtype=property_dtypes["unreduced_surface_area"]),
            "offsets": None,
        },
        "surface_thickness": {
            "values": np.zeros(n_endfeet, dtype=property_dtypes["surface_thickness"]),
            "offsets": None,
        },
    }

    # it is not guaranteed that endfoot index in consecutive
    for endfoot in endfeet:

        endfoot_index = endfoot.index

        properties["points"]["values"][endfoot_index] = endfoot.points
        properties["triangles"]["values"][endfoot_index] = endfoot.triangles

        properties["unreduced_surface_area"]["values"][endfoot_index] = endfoot.unreduced_area
        properties["surface_area"]["values"][endfoot_index] = endfoot.area
        properties["surface_thickness"]["values"][endfoot_index] = endfoot.thickness

    for name in ("points", "triangles"):

        properties[name]["offsets"][1:] = np.cumsum(
            [len(points) for points in properties[name]["values"]]
        )

        properties[name]["values"] = np.vstack(
            [points for points in properties[name]["values"] if len(points) > 0]
        ).astype(property_dtypes[name])

    export_grouped_properties(filename, properties)


def export_endfoot_mesh(endfoot_coordinates, endfoot_triangles, filepath):
    """Exports either all the faces of the laguerre cells separately or as one object
    in stl format"""
    import stl.mesh

    try:
        cell_mesh = stl.mesh.Mesh(np.zeros(len(endfoot_triangles), dtype=stl.mesh.Mesh.dtype))

        cell_mesh.vectors = endfoot_coordinates[endfoot_triangles]

        cell_mesh.save(filepath)

        L.info("Endfoot saved at: %s", filepath)

    except IndexError as e:
        msg = "No triangles found"
        L.error(msg)
        raise NGVError(msg) from e


def export_joined_endfeet_meshes(endfoot_iterator, filepath):
    """Exports the joined meshes TODO: fix this by replacing the endfoot iterato"""
    import stl.mesh

    vectors = np.array(
        [
            triangle.tolist()
            for endfoot in endfoot_iterator
            for triangle in endfoot.coordinates[endfoot.triangles]
        ]
    )

    cell_mesh = stl.mesh.Mesh(np.zeros(len(vectors), dtype=stl.mesh.Mesh.dtype))

    cell_mesh.vectors = vectors

    cell_mesh.save(filepath)
