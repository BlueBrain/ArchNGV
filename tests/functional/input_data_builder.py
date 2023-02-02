"""Script for building input test datasets to run
the ngv functional tests.
"""
import json
import logging
import math
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d
import openmesh
import sklearn
import sklearn.neighbors
import voxcell
from brainbuilder.app.atlases import (
    _build_hyperrectangle_brain_regions,
    _dump_atlases,
    _hyperrectangle_hierarchy,
    _normalize_hierarchy,
)
from cached_property import cached_property
from numpy import testing as npt
from priority_collections.priority_heap import MaxHeap
from scipy.spatial import cKDTree
from tqdm import tqdm
from vascpy import PointVasculature
from vascpy.conversion.graph_conversion import point_to_vasculature_data

from archngv.building.exporters.edge_populations import _write_edge_population
from archngv.core.datasets import Vasculature
from archngv.spatial.bounding_box import BoundingBox
from archngv.utils.binning import rebin_counts
from archngv.utils.geometry import unique_points

L = logging.getLogger(__name__)


def _unique_points_edges(points, edges):
    """Returns point connectivity with unique points and edges by collapsing
    duplicated coordinates and converting the respective edge indices
    """
    unique_idx, ps_to_uverts_map = unique_points(points, decimals=2)

    global_edges = ps_to_uverts_map[edges]

    # because vertices array has the same unique vertex id for duplicate coordinates
    # when we remapped the triangles we actually mapped to the unique index space.
    # Finally we remove the duplicate triangles via unique across rows after we make
    # sure that all the triangle ids are sorted
    _, edge_idx = np.unique(
        np.sort(global_edges, axis=1, kind="mergesort"), axis=0, return_index=True
    )

    # keep the initial order of the triangles
    # when selecting unique rows
    edge_idx.sort(kind="mergesort")
    return points[unique_idx], global_edges[edge_idx]


def _sample_elimination(points, rmax, fraction):
    """Implementation of the sample elimination algorithm, presented in
    the following paper: https://doi.org/10.1111/cgf.12538

    A fraction of the initial points are selected so that the final sample approximates
    poisson disk sampling. While in the paper the neighbors are calculated from a ball query of
    2rmax, here only the nearest neighbor is taken into account for performance reasons. In the
    end we need a sample that is more regularly distributed than the initial random sampling,
    not a super accurate poisson disk sampling behavior.

    The algorithm finds the nearest neighbor for each point and assigns a weight:
        wij = (1 - dij / 2rmax) ^ 8, with wij clipped in [0, 1]

    these weights are then used to populate a max priority heap, which sorts the point ids,
    with respect to the weight. Bigger the weight, the closest the neighbor is to that point.

    Popping from the heap, the point with the highest weight is removed and its neighbor is assigned


    Args:
        points (np.ndarray): Array of 3D points
        rmax (float): rmax that will be used in the algorithm
        fraction (float): The fraction of points that will be returned

    Returns:
        ids (np.ndarray): Array of int ids corresponding to the point subset
    """

    def _weights_and_neighbors(points, rmax):
        # returns a sparse csr_matrix
        graph = sklearn.neighbors.radius_neighbors_graph(
            points, 2.0 * rmax, mode="distance", include_self=False
        )

        # sanity check for matrix symmetry
        assert np.allclose(graph.data, graph.T.data)

        # update the non zero elements of the sparse matrix with the weight
        # computation specified in the paper
        graph.data = (1.0 - np.clip(0.5 * graph.data / rmax, 0.0, 1.0)) ** 8
        graph.data[np.isclose(graph.data, 0.0)] = 0.0
        graph.eliminate_zeros()

        return graph, graph.indptr, graph.indices

    graph, offs, inds = _weights_and_neighbors(points, rmax)

    # sum scipy matrix across axis 1 and return the result as an array
    weights = np.asarray(graph.sum(axis=1)).T[0]

    # create a max priority heap with an accuracy of 8 digits
    heap = MaxHeap(len(points), decimals=8)

    for i, weight in enumerate(weights):
        heap.push(i, weight)

    for _ in range(int((1.0 - fraction) * len(points))):
        pid, _ = heap.pop()

        # neighbors and their respective weights
        ns = inds[offs[pid] : offs[pid + 1]]
        ws = graph.data[offs[pid] : offs[pid + 1]]

        weights[ns] -= ws

        for nid, weight in zip(ns, weights[ns]):
            if not math.isclose(weight, 0.0):
                heap.update(nid, weight)

    return heap.ids


class Grid:
    """Grid region of interest. All datasets will be built using the grids bounding box.

    Args:
        shape (np.ndarray): int array with the number of voxels per dimension
        offset (np.ndarray): float array with the min point of the bbox
        voxel_side (float): the side of the voxel
    """

    @classmethod
    def from_cubic_bbox(cls, bbox_side, voxel_side, offset):
        """Creates a Grid from a cubic bbox"""
        extents = np.array([bbox_side, bbox_side, bbox_side])

        # find nearest integers so that the bbox is a multiple of the
        # voxel dimensions. We want an exact fit.
        shape = np.rint(extents / voxel_side).astype(np.int)

        return cls(shape.astype(np.int32), offset, voxel_side)

    def __init__(self, shape, offset, voxel_side):
        self.shape = shape
        self.offset = offset
        self.voxel_side = voxel_side

    @cached_property
    def voxel_dimensions(self):
        """Returns dimensions of grid voxels"""
        return np.array([self.voxel_side, self.voxel_side, self.voxel_side], dtype=np.float)

    @cached_property
    def voxel_volume(self):
        """Returns volume of grid voxels"""
        return np.prod(self.voxel_dimensions)

    @cached_property
    def lateral_area(self):
        """Returns the lateral area of the xz plane"""
        return np.prod(self.extents[[0, 2]])

    @property
    def min_point(self):
        """Returns the min point of the grid bbox"""
        return self.offset

    @cached_property
    def max_point(self):
        """Returns the max point of the grid bbox"""
        return self.offset + self.extents

    @cached_property
    def bbox(self):
        """Returns grid's bounding box"""
        return BoundingBox(self.min_point, self.max_point)

    @cached_property
    def centroid(self):
        """Returns the centroid of the grid bbox"""
        return 0.5 * (self.max_point + self.min_point)

    @cached_property
    def extents(self):
        """Returns the extents of the grid bbox"""
        return self.shape * self.voxel_dimensions

    @cached_property
    def volume(self):
        """Returns the volume of the grid bbox"""
        return np.prod(self.extents)

    def n_bins(self, axis):
        """Returns number of bins along axis"""
        return self.shape[axis]

    def n_bin_edges(self, axis):
        """Returns number of bin edges along axis"""
        return self.n_bins(axis) + 1

    def bins(self, axis):
        """Returns the grid binning across an axis"""
        return self.offset[axis] + np.arange(self.n_bin_edges(axis)) * self.voxel_side


class GridLayers:
    """Create layers aligned on the y direction of the input grid. The layers are normalized in [0., 1.]
    and their location depends on the y extent of the grid.
    """

    class Layers:
        """Helper class for cortical layers the location of which is normalized in [0, 1]
        with 0 being L6 and 1 being L1.
        """

        def __init__(self, ymin, ymax):
            normalized_bins = np.array(
                [0.0, 0.33627366, 0.58857402, 0.67966144, 0.84923722, 0.92074483, 1.0]
            )
            self._bins = ymin + (ymax - ymin) * normalized_bins
            self._labels = ["L6", "L5", "L4", "L3", "L2", "L1"]

        def __len__(self):
            """Returns number of layers"""
            return len(self._labels)

        @property
        def bins(self):
            """Returns layer bins"""
            return self._bins

        @property
        def labels(self):
            """Returns layer labels"""
            return self._labels

        def centers(self):
            """Returns bins centers"""
            bs = self.bins
            return 0.5 * (bs[1:] + bs[:-1])

        def thicknesses(self):
            """Returns layer thicknesses"""
            return np.diff(self.bins)

    def __init__(self, grid):
        self._grid = grid
        self.layers = self.Layers(self._grid.min_point[1], self._grid.max_point[1])

    def aligned_thicknesses(self, voxel_side):
        """Align layer boundaries along voxel grid."""
        result = []
        y0 = 0
        for y1 in np.cumsum(self.layers.thicknesses()):
            dy = voxel_side * max(1, np.round(y1) // voxel_side - y0 // voxel_side)
            y0 = y0 + dy
            result.append(dy)

        return np.asarray(result, dtype=np.float32)

    def build(self):
        """Build grid aligned layers to the grid"""
        labels = self.layers.labels
        thicknesses = self.aligned_thicknesses(self._grid.voxel_side)
        return OrderedDict(zip(labels[::-1], thicknesses[::-1]))


DENSITY_CONVERSION = {"um-3_to_mm-3": 1e9, "mm-3_to_um-3": 1e-9}


class GridBinnedDensity:
    """Binned density profile across the cortical depth with increasing bins."""

    @classmethod
    def create_from_n_astrocytes(cls, grid, n_astrocytes):
        """Create a density atlas, the density of which multiplied by its volume
        equals to n_astrocytes.
        """
        # the 0.1 is added to the n_astrocytes in order to make sure that we always
        # round down to the correct integer
        density = (float(n_astrocytes) + 0.1) / grid.volume
        densities = np.full(grid.n_bins(axis=1), fill_value=density, dtype=np.float32)

        return cls(grid, grid.bins(axis=1), densities)

    def __init__(self, grid, bins, densities):
        self._grid = grid
        self.density_bins = bins
        self.densities = densities
        self._density = None

    @property
    def density(self):
        """Returns density atlas if build, otherwise None"""
        return self._density

    def n_astrocytes(self):
        """Returns total number of astrocytes based on the density and the volume of the grid"""
        bin_lengths = np.diff(self.density_bins)
        return int(np.sum(bin_lengths * self.densities * self._grid.lateral_area))

    def align_bins(self):
        """Align bins at cortical surface (max value)"""
        self.density_bins += self._grid.bins(axis=1)[-1] - self.density_bins[-1]

    def match_grid_binning(self):
        """Rebin the density bins to match the grid bins"""
        bins = self._grid.bins(axis=1)

        lateral_volumes = np.diff(self.density_bins) * self._grid.lateral_area

        counts = self.densities * lateral_volumes
        new_counts = rebin_counts(counts, self.density_bins, bins)

        lateral_volumes = np.diff(bins) * self._grid.lateral_area

        self.density_bins = bins
        self.densities = new_counts / lateral_volumes

        npt.assert_equal(len(self.density_bins), len(self.densities) + 1)

    def build(self):
        """Build a density atlas for astrocytes"""

        if not (
            self._grid.n_bin_edges(axis=1) == self.density_bins.size
            and np.allclose(self._grid.bins(axis=1), self.density_bins)
        ):
            L.info("Density bins are not aligned. Aligning to grid bins...")
            self.align_bins()
            self.match_grid_binning()

        densities = np.empty(self._grid.shape, dtype=np.float32)

        for i in range(densities.shape[1]):
            densities[:, i, :] = self.densities[i]

        # density units mm-3
        densities *= DENSITY_CONVERSION["um-3_to_mm-3"]
        self._density = voxcell.VoxelData(
            densities, self._grid.voxel_dimensions, offset=self._grid.offset
        )

    def write(self, output_file):
        """Build density atlas and write it to file"""
        self._density.save_nrrd(str(output_file))

    def validate(self, filepath):
        """Validate output density atlas file. We want the density multiplied with the volume
        of the atlas to result to the number of astrocytes we want to place. Furthermore,
        the atlas dimensions are checked wrt to delimiting grid.
        """
        density = voxcell.VoxelData.load_nrrd(filepath)
        n_astrocytes = int(
            DENSITY_CONVERSION["mm-3_to_um-3"] * np.sum(density.raw * self._grid.voxel_volume)
        )

        npt.assert_array_equal(density.shape, self._grid.shape)
        npt.assert_allclose(density.offset, self._grid.min_point)
        npt.assert_allclose(density.shape * density.voxel_dimensions, self._grid.extents)
        npt.assert_equal(n_astrocytes, self.n_astrocytes())


class GridVasculature:
    """Synthetic space filling vascular tree embedded inside the boundaries
    of a 3D grid.
    """

    def __init__(self, grid):
        self._grid = grid
        self._vasculature = None

    @property
    def vasculature(self):
        """Returns vasculature object if it is built, None otherwise"""
        return self._vasculature

    @staticmethod
    def _level_tree(roots, basis):
        """Returns the vertices and edges of a space filling tree. For
        each root point it creates the orthant basis of the repeating
        tree pattern.

        Note: Duplicate points and edges are created. We deal with that later.

        Args:
            tuple:
                roots (List[np.ndarray]): Root
                basis (np.ndarray)
        """
        vertices = deepcopy(roots)
        edges = []

        offset = len(vertices)

        for i, root in enumerate(roots):
            r_vertices = root + basis
            n_vertices = len(r_vertices)

            vertices.extend(r_vertices)
            edges.extend((i, offset + j) for j in range(n_vertices))

            offset += n_vertices

        return vertices, edges

    def space_filling_tree(self, depth):
        """Create a  space filling tree based on the paper with doi
        (10.1109/IROS.2011.6094740)

        Args:
            min_point (np.ndarray): (3,) array of min bbox corner
            max_point (np.ndarray): (3,) array of max bbox corner
            level (int): The depth of the tree [0, inf]

        Returns:
            tuple:
                points (np.ndarray): (N, 3) points of tree vertices
                edges (np.ndarray): (M, 2) int array of tree edges
        """
        # unit cube corners (orthant basis)
        basis = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )

        # center the cube at (0,0,0), scale it by 0.5 to reflect the centers
        # of each orthant. Finally, this is scaled by the extent of the bbox
        # to reflect the bbox dimensions and not just a cube
        basis = 0.5 * (basis - 0.5) * self._grid.extents

        # first seed is the center of the bounding box
        vertices = [np.array([0.0, 0.0, 0.0])]

        # build the self-similar tree by recursively creating the orthant basis
        # using at each next step the vertices generated at the previous one and scaling
        # down the length of the basis by 0.5
        for i in range(depth):
            vertices, edges = GridVasculature._level_tree(vertices, basis * 0.5**i)

        # merge duplicated points and edges
        return _unique_points_edges(self._grid.centroid + np.vstack(vertices), np.vstack(edges))

    def _align_and_scale_to_grid(self, points):
        """Align created space filling tree points to reach the bounding box. While
        the space filling tree grows within the bbox, it doesn't touch its boundaries. Thus,
        points are scaled to achieve a tight fit.
        """
        bbox = BoundingBox.from_points(points)
        scale_factors = self._grid.extents / bbox.extent
        return scale_factors * (points - bbox.center) + self._grid.centroid

    def build(self, depth):
        """Create a space filling vascular tree with given depth. Diameters are uniformly
        sampled from [5, 8] microns.
        """
        points, edges = self.space_filling_tree(depth)

        # center tree points to grid center and scale so that they fit the grid box
        points = self._align_and_scale_to_grid(points)

        point_data = {"diameter": np.random.uniform(low=5.0, high=8.0, size=len(points))}
        edge_data = {"type": np.ones(len(edges), dtype=np.int)}

        node_properties, edge_properties = point_to_vasculature_data(
            points, edges, point_data, edge_data
        )

        self._vasculature = PointVasculature(node_properties, edge_properties)

    def write(self, output_file):
        """Build and write space filling vasculature to output file"""
        vasculature = self._vasculature.as_section_graph()
        vasculature.save(output_file)

    def validate(self, output_file):
        """Validate that the vasculature dataset fits the grid bbox"""
        vasculature = Vasculature.load(output_file)
        bbox = BoundingBox.from_points(vasculature.points)
        assert bbox == self._grid.bbox


class GridVasculatureMesh:
    """Mesh generator from a vasculature skeleton. Points are uniformly sampled
    on each segment and eliminated to approximate poisson disk sampling. Then a
    mesh is generated using the open3d library
    """

    def __init__(self, grid, samples_per_segment=1000):
        self._grid = grid

        self._n_samples = samples_per_segment

        phi_angles = np.random.uniform(0.0, 2.0 * np.pi, size=self._n_samples)

        # caching random values for perf. We don't care about bias
        self._cached_cosphi = np.cos(phi_angles)
        self._cached_sinphi = np.sin(phi_angles)
        self._cached_uniform = np.random.random(size=self._n_samples)

        self._vasculature_mesh = None

    @property
    def vasculature_mesh(self):
        """Returns the vasculature mesh object if built, otherwise None"""
        return self._vasculature_mesh

    def _unit_truncated_cone(self, r_min, r_max, length):
        """Distributes random points on the surface of the truncated cone and
        calculates their respective normals.
        """
        # 0% of the random points will be used for the caps
        n_caps = int(0.2 * self._n_samples)

        points = np.empty((self._n_samples, 3), dtype=np.float32)
        normals = np.empty_like(points)

        # initially we store at are the sqrt of the uniform random
        # variable to achieve uniform density on the disk
        r = np.sqrt(self._cached_uniform[:n_caps].copy())

        n_half = n_caps // 2

        # scale the unit radii for the two caps, assigned half the points for each
        r[:n_half] *= r_min
        r[n_half:] *= r_max

        points[:n_caps, 0] = r * self._cached_cosphi[:n_caps]
        points[:n_caps, 1] = r * self._cached_sinphi[:n_caps]

        points[:n_half, 2] = 0.0
        points[n_half:n_caps, 2] = length

        # cap normals pointing up or down
        normals[:n_caps, 0] = 0.0
        normals[:n_caps, 1] = 0.0
        normals[:n_half, 2] = -1.0
        normals[n_half:n_caps, 2] = 1.0

        sl_body = slice(n_caps, self._n_samples)

        # tip on the cone
        h_min = r_min * length / (r_max - r_min)

        # height of the truncated cone from its tip
        h_max = h_min + length

        # uniform sampling of truncated cone height, calculated from the integral of the probability
        # density on the conical surface.
        h = np.sqrt(self._cached_uniform[sl_body] * (h_max**2 - h_min**2) + h_min**2)

        # relationship from proportional right triangles
        r = h * r_max / h_max

        points[sl_body, 0] = r * self._cached_cosphi[sl_body]
        points[sl_body, 1] = r * self._cached_sinphi[sl_body]
        points[sl_body, 2] = h - h_min

        # normals calculated from the cross product of the partial derivatives
        # of the parametric equation above (rcosphi, rsinphi, r * h_max / r_max)
        normalization_constant = np.sqrt(h_max**2 + r_max**2)

        normals[sl_body, 0] = h_max * self._cached_cosphi[sl_body] / normalization_constant
        normals[sl_body, 1] = h_max * self._cached_sinphi[sl_body] / normalization_constant
        normals[sl_body, 2] = r_max / normalization_constant

        return points, normals

    @staticmethod
    def _rotate_z_axis_to_vector(u):
        """
        cross product of [0, 0, 1] with u
        v = [-u[1], u[0], 0.0]

        skew symmetric matrix K
        [  0   -v[2]  v[1] ]
        [ v[2]   0   -v[0] ]
        [-v[1]  v[0]   0   ]

        second power of skew symmetric matrix K ** 2
        [  -v[1] * v[1]    v[0] * v[1]   0  ]
        [   v[0] * v[1]   -v[2] * v[2]   0 ]
        [        0              0       - (v[0] * v[0] + v[1] * v[1]) ]

        rodriguez rotation formula
        R = I + K + K**2 (1 / (1 + dot([0, 0, 1], v)))
        """

        a = 1.0 / (1.0 + u[2])

        R = np.empty((3, 3))
        R[0, 0] = 1.0 - u[0] ** 2 * a
        R[0, 1] = -u[1] * u[0] * a
        R[0, 2] = u[0]
        R[1, 0] = R[0, 1]
        R[1, 1] = 1.0 - u[1] ** 2 * a
        R[1, 2] = u[1]
        R[2, 0] = -u[0]
        R[2, 1] = -u[1]
        R[2, 2] = 1.0 - (u[0] ** 2 + u[1] ** 2) * a

        return R

    def truncated_cone_sampling(self, p_min, p_max, r_min, r_max):
        """Uniform point sampling on the surface of the truncated cone."""
        vector = p_max - p_min
        length = np.linalg.norm(vector)
        u_vector = vector / length

        points, normals = self._unit_truncated_cone(r_min, r_max, length)

        R = GridVasculatureMesh._rotate_z_axis_to_vector(u_vector)

        points[:] = np.dot(points, R.T) + p_min
        normals[:] = np.dot(normals, R.T)

        return points, normals

    @staticmethod
    def _point_ids_outside_forks(v_points, v_diameters, v_degrees, points):
        """Remove points withint the intersection of the segments in a fork

        Returns:
            ids (np.ndarray): The ids of the points that are not inside the
            intersection of the segments.
        """
        mask = v_degrees > 1

        fork_points = v_points[mask]
        fork_radii = 1.1 * 0.5 * v_diameters[mask]

        t = cKDTree(points, copy_data=False)
        ids_set = set(
            index
            for index_list in t.query_ball_point(fork_points, fork_radii, eps=1e-5)
            for index in index_list
        )
        ids = np.fromiter((i for i in range(len(points)) if i not in ids_set), dtype=np.int)
        return ids

    def build_point_cloud(self, vasculature):
        """Create a point cloud distributed on the surface of the vasculature, determined"""
        L.info("Creating point sampling on vascular segments")

        mesh_data = np.empty((self._n_samples * vasculature.n_edges, 6), dtype=np.float32)

        v_points = vasculature.points
        v_diameters = vasculature.diameters
        radius = np.mean(0.5 * v_diameters)

        t = 0
        for v1, v2 in tqdm(vasculature.edges):
            r1, r2 = 0.5 * v_diameters[v1], 0.5 * v_diameters[v2]

            if r1 > r2:
                # swap points and radii
                v1, v2 = v2, v1
                r1, r2 = r2, r1

            seg_points, seg_normals = self.truncated_cone_sampling(
                v_points[v1], v_points[v2], r1, r2
            )

            ids = _sample_elimination(seg_points, radius, 0.1)

            n = ids.size
            mesh_data[t : t + n, :3] = seg_points[ids]
            mesh_data[t : t + n, 3:] = seg_normals[ids]

            t += n

        mesh_data = mesh_data[:t]
        points = mesh_data[:, :3]

        L.info("Removing points inside fork geometry...")
        ids = GridVasculatureMesh._point_ids_outside_forks(
            v_points, v_diameters, vasculature.degrees, points
        )
        mesh_data = mesh_data[ids]

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(mesh_data[:, :3])
        pcd.normals = open3d.utility.Vector3dVector(mesh_data[:, 3:])
        return pcd

    def build(self, vasculature):
        """Build vasculature's surface mesh using poisso point cloud reconstruction"""
        L.info("Building point cloud")
        point_cloud = self.build_point_cloud(vasculature)
        # open3d.visualization.draw_geometries([point_cloud])

        L.info("Creating mesh via poisson point cloud reconstruction")
        mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=7, scale=1.0, linear_fit=True
        )[0]

        L.info("Merging close vertices")
        mesh = mesh.merge_close_vertices(0.1)

        self._vasculature_mesh = mesh

    def write(self, output_file):
        """Write vasculature surface mesh"""
        open3d.io.write_triangle_mesh(output_file, self._vasculature_mesh)
        mesh = openmesh.read_trimesh(output_file)
        openmesh.write_mesh(output_file, mesh)

    def validate(self, output_file):
        """Validate surface mesh"""
        mesh = open3d.io.read_triangle_mesh(output_file)
        # open3d.visualization.draw_geometries([mesh])
        # assert mesh.is_orientable()
        # assert mesh.is_edge_manifold()
        # assert mesh.is_vertex_manifold()


class GridNeuronalCircuit:
    """Build a neuronal synthetic dataset (nodes, edges, config) embedded and bounded by a grid

    Args:
        n_neurons (int): Number of neurons in the dataset

    Notes:
        A realistic density of synapses 1 / um3 is created.
    """

    def __init__(self, grid, n_neurons, n_synapses):
        self.grid = grid

        self._n_neurons = n_neurons
        self._n_synapses = n_synapses

        L.info("Number of neurons %d, number of synapses %d", self._n_neurons, self._n_synapses)
        assert self._n_synapses % self._n_neurons == 0

    def build_neurons(self):
        """Build neuronal CellCollection"""
        cells = voxcell.CellCollection(population_name="All")
        cells.positions = np.random.uniform(
            self.grid.min_point, self.grid.max_point, size=(self._n_neurons, 3)
        )
        cells.properties["morphology"] = [f"neuron_{i}" for i in range(self._n_neurons)]
        cells.properties["mtype"] = "mock_mtype"
        cells.properties["model_type"] = "biophysical"

        return cells

    def write_neurons(self, output_file):
        """Write neurons sonata node population"""
        neurons = self.build_neurons()
        neurons.save_sonata(output_file)

    def write_synaptic_connectivity(self, output_file):
        """Write synaptic connectivity EdgePopulation"""

        ids = np.arange(self._n_synapses, dtype=np.uint)
        source_node_ids = ids % self._n_neurons
        target_node_ids = ids // self._n_neurons

        min_point = self.grid.min_point
        max_point = self.grid.max_point

        edge_properties = {
            "afferent_center_x": np.random.uniform(
                min_point[0], max_point[0], size=self._n_synapses
            ),
            "afferent_center_y": np.random.uniform(
                min_point[1], max_point[1], size=self._n_synapses
            ),
            "afferent_center_z": np.random.uniform(
                min_point[2], max_point[2], size=self._n_synapses
            ),
        }

        edge_properties["efferent_center_x"] = edge_properties["afferent_center_x"]
        edge_properties["efferent_center_y"] = edge_properties["afferent_center_y"]
        edge_properties["efferent_center_z"] = edge_properties["afferent_center_z"]

        _write_edge_population(
            str(output_file),
            source_population_name="All",
            target_population_name="All",
            source_population_size=self._n_neurons,
            target_population_size=self._n_neurons,
            source_node_ids=source_node_ids,
            target_node_ids=target_node_ids,
            edge_population_name="All",
            edge_properties=edge_properties,
        )

    @staticmethod
    def write_config(config_file, nodes_file, edges_file):
        """Write neuronal circuit sonata config"""

        config = {
            "manifest": {"$BASE_DIR": "."},
            "components": {"morphologies_dir": "NA"},
            "networks": {
                "nodes": [
                    {
                        "nodes_file": f"$BASE_DIR/{str(nodes_file.name)}",
                        "node_types_file": None,
                        "node_sets_file": "NA",
                        "id_offset": 1,
                    }
                ],
                "edges": [
                    {"edges_file": f"$BASE_DIR/{str(edges_file.name)}", "edge_types_file": None}
                ],
            },
        }

        with open(config_file, "w") as ofile:
            json.dump(config, ofile, indent=4)

    def write(self, nodes_output_file, edges_output_file, config_output_file):
        """Write neuronal circuit datasets"""
        self.write_neurons(nodes_output_file)
        self.write_synaptic_connectivity(edges_output_file)
        GridNeuronalCircuit.write_config(config_output_file, nodes_output_file, edges_output_file)

    def validate(self):
        pass


class GridBrainRegions:
    def __init__(self, grid):
        self.grid = grid

    def build_layers(self):
        return GridLayers(self.grid).build()

    def build(self):
        layers = self.build_layers()

        x_length, y_length, z_length = self.grid.extents
        brain_regions, region_ids = _build_hyperrectangle_brain_regions(
            x_length, z_length, layers, self.grid.voxel_side
        )
        return brain_regions, _normalize_hierarchy(_hyperrectangle_hierarchy(region_ids)), layers

    def write(self, output_dir):
        """Write atlases (brain_regions, hierarchy, layers)"""

        brain_regions, hierarchy, layers = self.build()
        _dump_atlases(brain_regions, layers, output_dir)

        with open(output_dir / "hierarchy.json", "w") as f:
            json.dump(hierarchy, f, indent=2)


class OutputPaths:
    """Output paths and directories"""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.atlas_dir = self.out_dir / "atlas"
        self.base_circuit_dir = self.out_dir / "circuit"

        self.base_circuit_config_path = self.base_circuit_dir / "circuit_config.json"
        self.base_circuit_nodes_path = self.base_circuit_dir / "nodes.h5"
        self.base_circuit_edges_path = self.base_circuit_dir / "edges.h5"
        self.density_path = self.atlas_dir / "[density]astrocytes.nrrd"
        self.vasculature_path = self.atlas_dir / "vasculature.h5"
        self.vasculature_mesh_path = self.atlas_dir / "vasculature.obj"

    def make_folder_hierarchy(self):
        """Create output folder hierarchy if it doesn't already exist"""
        self.out_dir.mkdir(exist_ok=True)
        L.info("Output directory: %s", self.out_dir)

        self.atlas_dir.mkdir(exist_ok=True)
        L.info("Atlas directory: %s", self.atlas_dir)

        self.base_circuit_dir.mkdir(exist_ok=True)
        L.info("Neuronal circuit directory: %s", self.base_circuit_dir)


def build_datasets(grid, paths, n_neurons, n_synapses, n_astrocytes):
    """Builds synthetic datasets of astrocytic density, neuronal node and edge
    populations, vasculature skeleton and surface mesh all bounded by the grid.

    Args:
        grid (Grid): blah
        paths (OutputPaths): bleh
        n_neurons (int):
        n_synapses (int):
        n_astrocytes (int):
    """
    # atlas density
    L.info("Building atlas density...")
    g_density = GridBinnedDensity.create_from_n_astrocytes(grid, n_astrocytes)
    g_density.build()
    g_density.write(paths.density_path)
    L.info("Validating atlas density...")
    g_density.validate(paths.density_path)

    # atlas regions, hierarchy, layers
    L.info("Building atlas regions, hierarchy and layers...")
    g_regions = GridBrainRegions(grid)
    g_regions.write(paths.atlas_dir)

    # neuronal circuit
    L.info("Building neuronal circuit...")
    L.info("Number of neurons: %d", n_neurons)
    L.info("Neuronal circuit sonata config output path: %s", paths.base_circuit_config_path)
    L.info("Neuronal circuit sonata nodes output path: %s", paths.base_circuit_nodes_path)
    L.info("Neuronal circuit sonata edges output path: %s", paths.base_circuit_edges_path)
    g_neurons = GridNeuronalCircuit(grid, n_neurons, n_synapses)
    g_neurons.write(
        paths.base_circuit_nodes_path, paths.base_circuit_edges_path, paths.base_circuit_config_path
    )

    # vasculature skeleton
    L.info("Building vasculature skeleton...")
    L.info("Vasculature skeleton output path: %s", paths.vasculature_path)
    g_vasculature = GridVasculature(grid)
    g_vasculature.build(depth=3)
    g_vasculature.write(paths.vasculature_path)
    g_vasculature.validate(paths.vasculature_path)

    # vasculature surface mesh
    L.info("Building vasculature surface mesh...")
    L.info("Vasculature surface mesh output path: %s", paths.vasculature_mesh_path)
    g_vasculature_mesh = GridVasculatureMesh(grid)
    g_vasculature_mesh.build(g_vasculature.vasculature)
    g_vasculature_mesh.write(str(paths.vasculature_mesh_path))
    g_vasculature_mesh.validate(str(paths.vasculature_mesh_path))


def run(out_dir):
    """Run synthetic input dataset building"""
    bbox_side = 70.0
    voxel_side = 7.0
    offset = np.array([0.0, 0.0, 0.0])

    # output paths
    paths = OutputPaths(out_dir)
    paths.make_folder_hierarchy()

    # make grid, the dimensions of which will constrain
    # all generated datasets
    grid = Grid.from_cubic_bbox(bbox_side, voxel_side, offset)

    n_astrocytes = 5
    n_neurons = 10
    n_synapses = 500
    n_synapses -= n_synapses % n_neurons

    build_datasets(
        grid, paths, n_neurons=n_neurons, n_synapses=n_synapses, n_astrocytes=n_astrocytes
    )


if __name__ == "__main__":
    import sys

    np.random.seed(0)
    run(Path(sys.argv[1]))
