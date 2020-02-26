""" Endfeet areas generation processing """
import logging

import numpy as np

# pylint: disable=no-name-in-module
from archngv.building.endfeet_reconstruction.fast_marching_method import fast_marching_eikonal_solver

from archngv.building.endfeet_reconstruction.area_mapping import transform_to_target_distribution
from archngv.building.endfeet_reconstruction.area_shrinking import shrink_surface_mesh

from archngv.building.endfeet_reconstruction.groups import vertex_to_triangle_groups
from archngv.building.endfeet_reconstruction.groups import group_elements

from archngv.utils.statistics import truncated_normal
from archngv.utils.ngons import vectorized_triangle_area
from archngv.utils.decorators import log_execution_time, log_start_end


L = logging.getLogger(__name__)


@log_start_end
@log_execution_time
def grow_endfeet_areas(vasculature_mesh, endfeet_points, threshold_radius):
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
        vasculature_mesh, endfeet_points, threshold_radius)
    return travel_times, group_indices


def _triangle_areas(points, triangles):
    """ Returns the areas of an array of triangles
    """
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
        The endfeet areas of the endeet

    Note:
        The difference between groups and endfeet indices is that the -1 group can
        be also present which coressponds to triangles that are not occupied by any
        endfoot.
    """
    endfeet_areas = np.zeros(n_endfeet, dtype=np.float)

    for group, ids in grouped_triangles.iter_assigned_groups():
        endfeet_areas[group] = triangle_areas[ids].sum()
    return endfeet_areas


def _shrink_endfoot_triangles(triangles, triangle_areas, triangle_travel_times, endfoot_area, target_area):
    """
    Shrinks the endfoot surface and convertes its triangles to the local index space
    """
    # indices of remaining t_ids and the remaining area
    idx = shrink_surface_mesh(triangle_areas, triangle_travel_times, endfoot_area, target_area)

    # remaining triangles
    t_tris = triangles[idx]

    vertices, inverse = np.unique(t_tris, return_inverse=True)

    # remap triangle indices to the local index space
    t_tris.ravel()[:] = np.arange(len(vertices))[inverse]

    return vertices, t_tris


def _process_endfeet(points, triangles, grouped_triangles,
                     triangle_areas, triangle_travel_times,
                     endfeet_areas, target_areas,
                     endfeet_thicknesses):
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

    Returns:
        Tuple generator containing the following data:
            - endfoot group id
            - points of endfoot surface mesh
            - triangles of endfoot surface mesh in the local index space
            - thickness of endfoot surface

    Note:
        The difference between group indices and endfoot indices is that groups
        include also the unassigned -1 group that corresponds to mesh triangles
        that are not occupid by endfeet.
    """
    for group, ids in grouped_triangles.iter_assigned_groups():

        current_area, target_area = endfeet_areas[group], target_areas[group]

        # endfoot area is overshoot, shrink it
        if current_area > target_area:

            t_verts, t_tris = _shrink_endfoot_triangles(
                triangles[ids],
                triangle_areas[ids],
                triangle_travel_times[ids],
                current_area, target_area)

            t_points = points[t_verts]
            final_area = _triangle_areas(t_points, t_tris).sum()

            yield group, t_points, t_tris, final_area, endfeet_thicknesses[group]


def endfeet_area_generation(vasculature_mesh, parameters, endfeet_points):
    """ Generate endfeet areas on the surface of the vasculature mesh,
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

    travel_times, vertex_groups = grow_endfeet_areas(
        vasculature_mesh, endfeet_points, parameters['fmm_cutoff_radius'])

    points = vasculature_mesh.points()
    triangles = vasculature_mesh.face_vertex_indices().astype(np.uintp)

    triangle_areas = _triangle_areas(points, triangles)

    # interpolate travel times at the center of triangles
    triangle_travel_times = np.mean(travel_times[triangles], axis=1)

    # find triangle groups from vertex groups
    triangle_groups = vertex_to_triangle_groups(vertex_groups, triangles)

    # convert the triangle groups into slices of unique group triangles
    grouped_triangles = group_elements(triangle_groups)

    # endfeet areas from the fast marching simulation
    endfeet_areas = _endfeet_areas(grouped_triangles, triangle_areas, n_endfeet)

    # input biological distribution of endfeet areas
    target_distribution = truncated_normal(*parameters['area_distribution'])

    # transformed simulation areas to map the target distribution for areas that have
    # spread more than the biological distribution
    target_areas = transform_to_target_distribution(endfeet_areas, target_distribution)

    # the thicknes that will be assigned to each endfoot
    endfeet_thicknesses = truncated_normal(*parameters['thickness_distribution']).rvs(size=n_endfeet)

    return _process_endfeet(points, triangles, grouped_triangles,
                            triangle_areas, triangle_travel_times,
                            endfeet_areas, target_areas, endfeet_thicknesses)
