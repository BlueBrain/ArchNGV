# SPDX-License-Identifier: Apache-2.0

""" Endfeet areas generation processing """
import logging

import numpy as np

from archngv.building.endfeet_reconstruction.area_mapping import transform_to_target_distribution
from archngv.building.endfeet_reconstruction.area_shrinking import shrink_surface_mesh

# pylint: disable=no-name-in-module
from archngv.building.endfeet_reconstruction.fast_marching_method import (
    fast_marching_eikonal_solver,
)
from archngv.building.endfeet_reconstruction.groups import group_elements, vertex_to_triangle_groups
from archngv.core.datasets import EndfootMesh
from archngv.utils.ngons import vectorized_triangle_area
from archngv.utils.statistics import truncated_normal

L = logging.getLogger(__name__)


def _grow_endfeet_meshes(vasculature_mesh, endfeet_points, threshold_radius):
    """
    Args:
        mesh: TriMesh
            Vasculature mesh
        endfeet_points: array[float, 3]
            The coordinates of the endfeet contacts on the surface
            of the vasculature.
        max_area: float
            Maximum permitted area for the growht of the endfeet.
    """
    group_indices, travel_times, _ = fast_marching_eikonal_solver(
        vasculature_mesh, endfeet_points, threshold_radius
    )
    return travel_times, group_indices


def _triangle_areas(points, triangles):
    """Returns the areas of an array of triangles"""
    p0s, p1s, p2s = points[triangles.T]
    return vectorized_triangle_area(p0s - p1s, p0s - p2s)


def _endfeet_areas(grouped_triangles, triangle_areas, n_endfeet):
    """
    Args:
        grouped_triangles:
            Array of triangle ids
        triangle_areas:
            Areas of triangles
        n_endfeet:
            Total number of endfeet

    Returns:
        The areas of the endfeet.

    Note:
        The difference between groups and endfeet indices is that the -1 group can
        be also present which coressponds to triangles that are not occupied by any
        endfoot.
    """
    endfeet_areas = np.zeros(n_endfeet, dtype=np.float32)

    for group, ids in grouped_triangles.iter_assigned_groups():
        endfeet_areas[group] = triangle_areas[ids].sum()

    return endfeet_areas


def _global_to_local_indices(triangles):
    local_vertices, inverse = np.unique(triangles, return_inverse=True)

    # remap triangle indices to the local index space
    raveled_triangles = np.arange(len(local_vertices))[inverse]

    return local_vertices, raveled_triangles.reshape((-1, 3))


def _shrink_endfoot_triangles(
    triangles, triangle_areas, triangle_travel_times, endfoot_area, target_area
):
    """
    Shrinks the endfoot surface and convertes its triangles to the local index space
    """
    # indices of remaining t_ids and the remaining area
    idx = shrink_surface_mesh(triangle_areas, triangle_travel_times, endfoot_area, target_area)
    return triangles[idx]


def _process_endfeet(
    points,
    triangles,
    grouped_triangles,
    triangle_areas,
    triangle_travel_times,
    endfeet_areas,
    target_areas,
    endfeet_thicknesses,
):
    """
    Iterates over the grown endfeet surfaces and shrinks them so that
    they match their respective target areas.

    Args:
        points: np.ndarray (N, 3)
            All point of the vasculature mesh

        triangles: np.ndarray (M, 3)
            All triangles of the vasculature mesh

        group_triangles: GroupedTriangles
            Endfeet groups with their respective triangle slices

        triangle_areas,
            All triangle areas: np.ndarray (M,)

        triangle_travel_times: np.ndarray (M, )
            Interpolated travel times from the vertices to their triangles

        endfeet_areas: np.ndarray (K,)
            The total areas of the endfeet

        target_areas: np.ndarray (K, )
            The target areas of the endfeet that we desire

        endfeet_thicknesses: np.ndarray (K,)
            The thickness of the endfoot surface mesh

    Yields:
        EndfootMesh data object.

    Note:
        The difference between group indices and endfoot indices is that groups
        include also the unassigned -1 group that corresponds to mesh triangles
        that are not occupid by endfeet.

        Only the groups that have grown are yielded. Therefore, gaps are possible
        but the results are yielded in a sorted incremental manner.
    """
    for group, triangle_ids in grouped_triangles.iter_assigned_groups():
        current_area, target_area = endfeet_areas[group], target_areas[group]

        # triangles for endfoot but the indices are from the entire mesh
        triangles_global = triangles[triangle_ids]

        if current_area > target_area:
            triangles_global = _shrink_endfoot_triangles(
                triangles_global,
                triangle_areas[triangle_ids],
                triangle_travel_times[triangle_ids],
                current_area,
                target_area,
            )

        # the unique vertices belonging to group and the triangles referring to that subset
        vertices_global, triangles_local = _global_to_local_indices(triangles_global)

        points_local = points[vertices_global]
        final_area = _triangle_areas(points_local, triangles_local).sum()

        yield EndfootMesh(
            index=group,
            points=points_local,
            triangles=triangles_local,
            area=final_area,
            unreduced_area=current_area,
            thickness=endfeet_thicknesses[group],
        )


def endfeet_area_generation(vasculature_mesh, parameters, endfeet_points):
    """Generate endfeet areas on the surface of the vasculature mesh,
    starting fotm the endfeet_points coordinates

    Args:
        vasculature_mesh: Trimesh
            The mesh of the vasculature

        parameters: dict
            The parameters for the algorithms with the following keys:
                - area_distribution [mean, sdev, min, max]
                - thickness_distribution [mean, sdev, min, max]

        endfeet_points: ndarray (N, 3)
            Endfeet target coordinates
    """
    n_endfeet = len(endfeet_points)

    travel_times, vertex_groups = _grow_endfeet_meshes(
        vasculature_mesh, endfeet_points, parameters["fmm_cutoff_radius"]
    )

    points = vasculature_mesh.points()
    triangles = vasculature_mesh.face_vertex_indices().astype(np.uintp)

    triangle_areas = _triangle_areas(points, triangles)

    # interpolate travel times at the center of triangles
    triangle_travel_times = np.mean(travel_times[triangles], axis=1)

    # for each triangle deduce its group id by examining its vertices
    triangle_groups = vertex_to_triangle_groups(vertex_groups, triangles)

    # make chunks of triangles that belong to the same group
    grouped_triangles = group_elements(triangle_groups)

    # endfeet areas from the fast marching simulation
    endfeet_areas = _endfeet_areas(grouped_triangles, triangle_areas, n_endfeet)

    # input biological distribution of endfeet areas
    target_distribution = truncated_normal(*parameters["area_distribution"])

    # transformed simulation areas to map the target distribution for areas that have
    # spread more than the biological distribution
    target_areas = transform_to_target_distribution(endfeet_areas, target_distribution)

    # the thickness that will be assigned to each endfoot
    endfeet_thicknesses = truncated_normal(*parameters["thickness_distribution"]).rvs(
        size=n_endfeet
    )

    return _process_endfeet(
        points,
        triangles,
        grouped_triangles,
        triangle_areas,
        triangle_travel_times,
        endfeet_areas,
        target_areas,
        endfeet_thicknesses,
    )
