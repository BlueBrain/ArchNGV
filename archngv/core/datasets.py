# SPDX-License-Identifier: Apache-2.0

"""Archngv dataset classes."""
import collections.abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union

import h5py
import numpy as np
from cached_property import cached_property

from archngv.core.sonata_readers import EdgesReader, NodesReader
from archngv.exceptions import NGVError
from archngv.spatial import ConvexPolygon

DOMAIN_TRIANGLE_TYPE: Dict[str, Union[int, slice]] = {"polygon_id": 0, "vertices": slice(1, 4)}


class CellData(NodesReader):
    """Cell population information"""

    def positions(self, index=None):
        """Cell positions"""
        return self.get_properties(["x", "y", "z"], ids=index)


class GliovascularConnectivity(EdgesReader):
    """Access to the Gliovascular Data"""

    def astrocyte_endfeet(self, astrocyte_ids):
        """endfoot_id is equivalent to the edge id. Can resolve quicker using afferent_edges"""
        return self.afferent_edges(astrocyte_ids)

    def vasculature_surface_targets(self, endfeet_ids=None):
        """Endfeet surface targets on vasculature."""
        return self.get_properties(
            ["endfoot_surface_x", "endfoot_surface_y", "endfoot_surface_z"],
            ids=endfeet_ids,
        )

    def vasculature_sections_segments(self, endfeet_ids):
        """Returns the edge id, morphology section and segment id for each endfoot"""
        edge_ids = self.get_source_nodes(endfeet_ids)
        efferent = self.get_properties(
            ["vasculature_section_id", "vasculature_segment_id"], ids=endfeet_ids
        )
        return np.column_stack((edge_ids, efferent))


class NeuronalConnectivity(EdgesReader):
    """Synaptic data access"""

    def synapse_positions(self, synapse_ids=None):
        """XYZ coordinates for given synapse_ids (all if synapse_ids not specified)"""
        syn_positions = [
            ["efferent_center_x", "efferent_center_y", "efferent_center_z"],
            ["afferent_center_x", "afferent_center_y", "afferent_center_z"],
        ]

        for position_properties in syn_positions:
            try:
                return self.get_properties(position_properties, synapse_ids)
            except NGVError:
                pass

        raise NGVError(f"Cannot find positions inside {self.filepath}")

    def target_neurons(self, synapse_ids=None):
        """Target neuron's node ids for given synapse_ids."""
        return self._impl.target_nodes(self._selection(synapse_ids))

    @cached_property
    def target_neuron_count(self):
        """Number of unique target neurons."""
        return len(np.unique(self.target_neurons()))


class NeuroglialConnectivity(EdgesReader):
    """Neuroglial connectivity access."""

    def astrocyte_neuron_connections(self, astrocyte_id):
        """Returns edge ids between astrocyte and neurons"""
        return self.efferent_edges(astrocyte_id)

    def neuronal_synapses(self, connection_ids):
        """Returns the synapse ids"""
        return self.get_property("synapse_id", ids=connection_ids)

    def astrocyte_synapses(self, astrocyte_id):
        """Get the synapse ids connected to a given `astrocyte_id`."""
        edge_ids = self.astrocyte_neuron_connections(astrocyte_id)
        return self.neuronal_synapses(edge_ids)

    def astrocyte_number_of_synapses(self, astrocyte_id):
        """Get the number of synapses to a given `astrocyte_id`."""
        return len(np.unique(self.astrocyte_synapses(astrocyte_id)))

    def astrocyte_neurons(self, astrocyte_id, unique=True):
        """Post-synaptic neurons given an `astrocyte_id`."""
        return self.efferent_nodes(astrocyte_id, unique=unique)


class GlialglialConnectivity(EdgesReader):
    """Glialglial connectivity access."""

    def astrocyte_astrocytes(self, astrocyte_id, unique=True):
        """Target astrocyte connected to astrocyte with `astrocyte_id`."""
        return self.efferent_nodes(astrocyte_id, unique=unique)


class Microdomain(ConvexPolygon):
    """Extends Convex Polygon shape into an astrocytic microdomain with extra properties"""

    def __init__(self, points: np.ndarray, triangle_data: np.ndarray, neighbors: np.ndarray):
        self._polygon_ids = triangle_data[:, DOMAIN_TRIANGLE_TYPE["polygon_id"]]
        triangles = triangle_data[:, DOMAIN_TRIANGLE_TYPE["vertices"]]
        super().__init__(points, triangles)
        self.neighbor_ids = neighbors

    @property
    def polygons(self) -> List[List[int]]:
        """Returns the polygons of the domain"""
        from archngv.utils.ngons import triangles_to_polygons

        return triangles_to_polygons(self._triangles, self._polygon_ids)

    @property
    def triangle_data(self) -> np.ndarray:
        """Returns the triangle data of the microdomains
        [polygon_id, v0, v1, v2]
        """
        return np.column_stack((self._polygon_ids, self._triangles))

    def scale(self, scale_factor: float) -> "Microdomain":
        """Uniformly scales the polygon by a scale_factor, assuming its centroid
        sits on the origin.
        """
        cnt = self.centroid
        return Microdomain(
            scale_factor * (self.points - cnt) + cnt,
            self.triangle_data.copy(),
            self.neighbor_ids.copy(),
        )


class H5ContextManager:
    """Context manager for hdf5 files"""

    def __init__(self, filepath):
        self._fd = h5py.File(filepath, "r")

    def close(self):
        """Close hdf5 file"""
        self._fd.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """And exit"""
        self.close()


class GroupedProperties(H5ContextManager):
    """Access grouped properties in an hdf5 file

    Properties are stored at the root level and they are accompanied by an offset dataset with the
    same name as the property. (e.g. /points and /offsets/points)

    Example:

    /property1
    /property2
    /offsets/property1
    /offsets/property2

    The values that correspond in each group can be accessed via the respective offsets. The values
    in the i-th group correspond to values[offsets[i]: offsets[i + 1]].
    """

    @property
    def _offsets(self) -> h5py.Group:
        """Returns the offsets group"""
        return self._fd["offsets"]

    @property
    def _data(self) -> h5py.Group:
        """Returns the data group"""
        return self._fd["data"]

    def _offset_slice(self, property_name, group_index) -> Tuple[int, int]:
        """Returns the slice of offset_type indices (beg, end) for astrocyte_index"""
        return self._offsets[property_name][group_index : group_index + 2]

    def __len__(self) -> int:
        """Returns the number of properties by checking the size of the first dataset in offsets."""
        first_offset_name = next(iter(self._offsets.keys()))
        return len(self._offsets[first_offset_name]) - 1

    @property
    def property_names(self) -> List[str]:
        """Returns all available properties in the dataset"""
        return list(self._data)

    def _get_data_slice(self, property_name: str, group_index: int) -> np.ndarray:
        """Returns the data slice corresponding to the group index"""
        beg, end = self._offset_slice(property_name, group_index)

        return self._data[property_name][beg:end]

    def get(self, property_name: str, group_index: Optional[int] = None) -> Any:
        """
        Returns the values of property with `property_name` at `group_index` if not None, else
            all the available values.

        Args:
            property_name: The name of the property to retrieve.
            group_index: The index of the group of values to retrieve, if any. Default is None,
                in which case all values are returned.

        Returns:
            A single value (group_index not None) or a numpy array of the queried values, which
                can be multi-dimensional.

        Note:
            The group_index is cast into a python int, to avoid numpy's uint64 + int -> float64
            type conversion.
        """
        if group_index is None:
            return self._data[property_name][:]

        group_index = int(group_index)

        # no offsets -> one property per group
        if property_name not in self._offsets:
            return self._data[property_name][group_index]

        return self._get_data_slice(property_name, group_index)


TObject = TypeVar("TObject")

OneOrList = Union[TObject, List[TObject]]
OneOrIterable = Union[TObject, Iterable[TObject]]


def _apply_callable(
    func: Callable[[int], TObject], key: OneOrIterable[np.integer]
) -> OneOrList[TObject]:
    """
    Resolves the key to either apply the callable function at a single index or an iterable of
        indices.

    Args:
        func: A function that take an integer and returns an object.
        key: An integer-like or iterable of integer-like numbers.

    Returns:
        Either a single object corresponding to the integer key or a list of objects corresponding
        to the iterable of integers.

    Note: The key is casted to int, to avoid the unint64 + int -> float64 error
    """
    if isinstance(key, collections.abc.Iterable):
        return [func(int(i)) for i in iter(key)]

    # covers both python int and numpy integer types
    if np.issubdtype(type(key), np.integer):
        return func(int(key))

    raise TypeError(
        f"Incompatible key type. Expected integer-like or iterable of integer-likes, got {key}"
    )


class Microdomains(GroupedProperties):
    """Data structure for storing the information concerning the microdomains."""

    def __iter__(self) -> Iterator[Microdomain]:
        """Microdomain object iterator."""
        for i in range(self.n_microdomains):
            yield self.domain_object(i)

    def __getitem__(self, key: OneOrIterable[np.integer]) -> OneOrList[Microdomain]:
        """List getter."""
        return _apply_callable(self.domain_object, key)

    @property
    def n_microdomains(self) -> int:
        """Total number of Microdomains."""
        return len(self)

    def n_neighbors(self, astrocyte_index: int, omit_walls: bool = True) -> int:
        """Number of neighboring microdomains around microdomains using astrocyte_index."""
        return len(self.domain_neighbors(astrocyte_index, omit_walls=omit_walls))

    def domain_neighbors(self, astrocyte_index: int, omit_walls: bool = True) -> np.ndarray:
        """For every triangle in the microdomain return its respective neighbor.

        Multiple triangles can have the same neighbor if the are part of a triangulated
        polygon face. A microdomain can also have a bounding box wall as a neighbor
        which is signified with a negative number.
        """
        neighbors = self.get("neighbors", astrocyte_index)
        if omit_walls:
            return neighbors[neighbors >= 0]
        return neighbors

    def domain_is_boundary(self, astrocyte_index: int) -> np.bool_:
        """Returns true if the domain is adjacent to a wall.
        The walls have negative indices as "neighbors"
        """
        return np.any(self.get("neighbors", astrocyte_index) < 0)

    def domain_points(self, astrocyte_index: int) -> np.ndarray:
        """The coordinates of the vertices of the microdomain."""
        return self.get("points", astrocyte_index)

    def domain_triangle_data(self, astrocyte_index: int) -> np.ndarray:
        """Returns the triangle data of the tessellation.

        Returns:
            numpy.ndarray: [polygon_id, v0, v1, v2]
        """
        return self.get("triangle_data", astrocyte_index)

    def domain_triangles(self, astrocyte_index: int) -> np.ndarray:
        """Returns the triangles connectivity of the domain_points from an astrocyte."""
        triangle_data = self.domain_triangle_data(astrocyte_index)
        return triangle_data[:, DOMAIN_TRIANGLE_TYPE["vertices"]]

    def domain_object(self, astrocyte_index: int) -> Microdomain:
        """Returns a archngv.core.dataset Microdomain object."""
        return Microdomain(
            self.domain_points(astrocyte_index),
            self.domain_triangle_data(astrocyte_index),
            self.domain_neighbors(astrocyte_index, omit_walls=False),
        )

    @cached_property
    def connectivity(self) -> np.ndarray:
        """Returns the connectivity of the microdomains."""
        edges = [
            (cid, nid)
            for cid in range(self.n_microdomains)
            for nid in self.domain_neighbors(cid, omit_walls=True)
        ]
        # sort by column [2 3 1] -> [1 2 3]
        sorted_by_column = np.sort(edges, axis=1)
        # take the unique rows
        return np.unique(sorted_by_column, axis=0)

    def global_triangles(self) -> np.ndarray:
        """Converts microdomain tessellation to a joined mesh.

        Converts the per object tessellation to a joined mesh with unique points and triangles
        of unique vertices.

        Returns:
            points: array[float, (N, 3)]
            triangles: array[int, (M, 3)]
            neighbors: array[int, (M, 3)]
        """
        from archngv.utils.ngons import local_to_global_mapping

        ps_tris_offsets = np.column_stack(
            (self._offsets["points"][:], self._offsets["triangle_data"][:])
        )

        return local_to_global_mapping(
            self.get("points"),
            self.get("triangle_data")[:, DOMAIN_TRIANGLE_TYPE["vertices"]],
            ps_tris_offsets,
        )

    def global_polygons(self) -> Tuple[np.ndarray, List[List[int]]]:
        """Returns unique points and polygons in the global index space."""
        from archngv.utils.ngons import (
            local_to_global_mapping,
            local_to_global_polygon_ids,
            triangles_to_polygons,
        )

        triangle_data = self.get("triangle_data")

        g_poly_ids = local_to_global_polygon_ids(
            triangle_data[:, DOMAIN_TRIANGLE_TYPE["polygon_id"]]
        )

        ps_tris_offsets = np.column_stack(
            (self._offsets["points"][:], self._offsets["triangle_data"][:])
        )

        # local to global triangles
        ps, tris, polys = local_to_global_mapping(
            self.get("points"),
            triangle_data[:, DOMAIN_TRIANGLE_TYPE["vertices"]],
            ps_tris_offsets,
            triangle_labels=g_poly_ids,
        )

        return ps, triangles_to_polygons(tris, polys)

    def export_mesh(self, filename: Path) -> None:
        """Write the tessellation to file as a mesh."""
        import stl.mesh

        points, triangles = self.global_triangles()
        cell_mesh = stl.mesh.Mesh(np.zeros(len(triangles), dtype=stl.mesh.Mesh.dtype))
        cell_mesh.vectors = points[triangles]
        cell_mesh.save(filename)


@dataclass
class EndfootMesh:
    """Endfoot mesh data class"""

    index: int
    points: np.ndarray
    triangles: np.ndarray
    area: np.float32
    unreduced_area: np.float32
    thickness: np.float32


class EndfootSurfaceMeshes(GroupedProperties):
    """Access to the endfeet meshes."""

    def _object(self, index: int) -> EndfootMesh:
        return EndfootMesh(
            index=index,
            points=self.mesh_points(index),
            triangles=self.mesh_triangles(index),
            area=self.get("surface_area", index),
            unreduced_area=self.get("unreduced_surface_area", index),
            thickness=self.get("surface_thickness", index),
        )

    def __getitem__(self, key: OneOrIterable[np.integer]) -> OneOrList[EndfootMesh]:
        """Endfoot mesh object getter."""
        return _apply_callable(self._object, key)

    def __iter__(self) -> Iterator[EndfootMesh]:
        """Endfoot iterator."""
        for index in range(self.__len__()):
            yield self._object(index)

    def mesh_points(self, endfoot_index: Optional[int] = None) -> np.ndarray:
        """Return the points of the endfoot mesh."""
        return self.get("points", group_index=endfoot_index)

    def mesh_triangles(self, endfoot_index: Optional[int] = None) -> np.ndarray:
        """Return the triangles of the endfoot mesh."""
        return self.get("triangles", group_index=endfoot_index)
