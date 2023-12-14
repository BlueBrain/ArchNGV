""" Functions to reduce the area of surface meshes using
travel time from fmm simulation
"""
import numpy as np


def shrink_surface_mesh(triangle_areas, triangle_travel_times, current_area, target_area):
    """
    Args:
        triangle_areas: (N, 3)
            Array of triangle arrays to filter

        triangle_travel_times: (N,)
            Array of travel times that correspond to the triangles

        current_area:
            Total area of the triangle_areas

        target_area:
            Total area that we need to reach

    Note:
        It should hold that current_area > target_area

    Returns:
        Indices of triangles to keep
    """
    area_to_remove = current_area - target_area

    descending_idx = np.argsort(triangle_travel_times, kind="stable")[::-1]

    # sort areas wrt to the travel times
    cumulative_areas = np.cumsum(triangle_areas[descending_idx])

    start_pos = np.searchsorted(cumulative_areas, area_to_remove) + 1

    return descending_idx[start_pos:]
