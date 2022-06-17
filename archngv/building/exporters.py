"""SONATA node and edge population exporters"""
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

import h5py
import libsonata
import numpy as np
import voxcell

from archngv.core.datasets import EndfootMesh, Microdomain
from archngv.exceptions import NGVError

L = logging.getLogger(__name__)


def add_properties_to_edge_population(
    filepath: Path, population_name: str, properties: Dict[str, np.ndarray]
) -> None:
    """Add properties that are not already existing to an edge population.

    Args:
        filepath: SONATA EdgePopulation h5 file path
        population_name: The name of the EdgePopulation
        properties: A dict with property names as keys and 1D numpy arrays as values.

    Raises:
        AssertionError: If property name exists or if property values length is not
            compatible with the edge population
    """
    with h5py.File(filepath, "r+") as h5f:

        group = h5f[f"/edges/{population_name}/0"]
        length = h5f[f"/edges/{population_name}/source_node_id"].shape[0]

        for name, values in properties.items():

            if name in group:
                raise NGVError(f"'{name}' property already exists.")

            if values.size != length:
                raise NGVError(f"Incompatible length. Expected: {length}. Given: {values.size}")

            group.create_dataset(name, data=values)
            L.info("Added edge Property: %s", name)


def write_edge_population(
    output_path: Path,
    population_name: str,
    source_population: voxcell.CellCollection,
    target_population: voxcell.CellCollection,
    source_node_ids: np.ndarray,
    target_node_ids: np.ndarray,
    properties: Dict[str, np.ndarray],
) -> None:
    """Write a SONATA node population.

    Args:
        output_path: Filepath to write.
        population_name: Name of edge population.
        source_population: The source SONATA nodes, opened with voxcell.CellCollection.
        target_population: The target SONATA nodes, opened with voxcell.CellCollection.
        source_node_ids: The ids of the source node population.
        target_node_ids: The ids of the target node population.
        properties: The dictionary of properties to write to the population.
    """
    # pylint: disable=too-many-arguments

    assert len(source_node_ids) == len(target_node_ids)

    with h5py.File(output_path, "w") as h5f:
        h5root = h5f.create_group(f"/edges/{population_name}")

        # 'edge_type_id' is a required attribute storing index into CSV which we don't use
        h5root.create_dataset(
            "edge_type_id", data=np.full(len(source_node_ids), -1, dtype=np.int32)
        )

        h5root.create_dataset("source_node_id", data=source_node_ids, dtype=np.uint64)
        h5root.create_dataset("target_node_id", data=target_node_ids, dtype=np.uint64)

        h5group = h5root.create_group("0")

        # add edge properties
        for name, values in properties.items():
            h5group.create_dataset(name, data=values)
            L.info("Added edge Property: %s", name)

        h5root["source_node_id"].attrs["node_population"] = source_population.population_name
        h5root["target_node_id"].attrs["node_population"] = target_population.population_name

    if len(source_node_ids) > 0:
        L.info("Creating edge indexing in: %s", population_name)
        # above, edge population has been sorted by (target_id, source_id)
        libsonata.EdgePopulation.write_indices(
            output_path,
            population_name,
            source_node_count=len(source_population),
            target_node_count=len(target_population),
        )
    else:
        L.warning("Indexing will not be done. No edges in: %s", population_name)


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


def export_microdomains(
    filename: Path, domains: Iterable[Microdomain], scaling_factors: np.ndarray
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
