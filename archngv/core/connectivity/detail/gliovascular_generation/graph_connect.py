""" Connect astrocytes with vasculature targets
"""

import logging

import numpy
import scipy.stats
import scipy.sparse

from morphspatial import inclusion
from morphspatial import collision


L = logging.getLogger(__name__)


def _num_endfeet_distribution(mean, std, clip_a, clip_b):
    """ Returns the truncated normal distribution with (mean, std)
    minimum value clip_a and maximum value clip_b
    """
    a, b = (clip_a - mean) / std, (clip_b - mean) / std

    return scipy.stats.truncnorm(a, b, mean, std)


def _radius_approx_from_segments(vasculature, target_segment_idx):
    """ Aprroximate the radius of the target point by averaging
    the ends of the segment it belongs in. The error is not high
    due to the short length of segments compared to their radii.
    """
    seg_radii_begs, seg_radii_ends = vasculature.segments_radii

    target_radii = \
        0.5 * (seg_radii_begs[target_segment_idx] + seg_radii_ends[target_segment_idx])

    return target_radii


def points_inside_bounding_box(bbox, target_positions):
    """
    Args:
        bbox: tuple [xmin, ymin, zmin, xmax, ymax, zmax]
        positions: array[float, (N, 3)]

    Returns: array[bool, (N,)]
        True if point inside bounding box
    """
    return (bbox[0] <= target_positions[:, 0])  & (target_positions[:, 0] <= bbox[3]) & \
           (bbox[1] <= target_positions[:, 1])  & (target_positions[:, 1] <= bbox[4]) & \
           (bbox[2] <= target_positions[:, 2])  & (target_positions[:, 2] <= bbox[5])


def _find_available_targets(domain_shape, bb_idx, target_positions, target_radii):


    # spheres inside domain are accepted without checks
    center, radius = domain_shape.inscribed_sphere

    assert radius > 0.

    mask_inscribed_sphere = \
    inclusion.spheres_in_sphere(target_positions[bb_idx], target_radii[bb_idx], center, radius)

    # now we need to check the spheres that are outside
    # the inscribed sphere, and inside the bounding box
    masked_idx = bb_idx[~mask_inscribed_sphere]

    # check actual intersections with geometry of domain
    domain_face_points = domain_shape.points[domain_shape.triangles[:, 0]]

    mask_intersecting = collision.convex_shape_with_spheres(domain_face_points,
                                                            domain_shape.face_normals,
                                                            target_positions[masked_idx],
                                                            target_radii[masked_idx])

    if mask_intersecting.any():

        targets_idx = numpy.hstack((masked_idx[mask_intersecting], bb_idx[mask_inscribed_sphere]))

    else:

        targets_idx = bb_idx[mask_inscribed_sphere]

    return targets_idx


def _filter_according_to_strategy(domain_position,
                                  target_positions,
                                  target_radii,
                                  number_of_endfeet,
                                  reachout_strategy_function):

    n_targets = len(target_positions)

    if number_of_endfeet < n_targets:

        effective_distances = \
        numpy.linalg.norm(domain_position - target_positions, axis=1)

        effective_distances -= target_radii

        if number_of_endfeet == 1:

            idx = numpy.array([[numpy.argmin(effective_distances)]])

        else:

            idx = reachout_strategy_function(effective_distances, number_of_endfeet)

    else:

        idx = numpy.arange(n_targets, dtype=numpy.intp)

    return idx


def domains_to_vasculature(cell_ids,
                           vasculature,
                           reachout_strategy_function,
                           target_positions,
                           target_vasculature_segments,
                           microdom,
                           properties):
    """
    1. Generate structural connectivity from the geometrical aspects
    of hulls and target spheres.

    2. For each domain determine the number of endfeet according to a biological
    normal distribution

    3. Using a reachout strategy select twhich ones to keep

    """
    target_radii = _radius_approx_from_segments(vasculature,
                                                target_vasculature_segments)

    L.info('Endfeet Distribution Paremeters %s', properties['Endfeet Distribution'])

    n_distr = _num_endfeet_distribution(*properties['Endfeet Distribution'])

    domain_target_edges = []

    idx = numpy.arange(len(target_positions), dtype=numpy.intp)

    for domain_index, cell_id in enumerate(cell_ids):

        number_of_endfeet = numpy.round(n_distr.rvs()).astype(numpy.int)

        if number_of_endfeet == 0:
            continue

        domain_shape = microdom.domain_object(int(cell_id))
        mask_bb = points_inside_bounding_box(domain_shape.bounding_box, target_positions)

        if not mask_bb.any():
            continue

        targets_idx = \
            _find_available_targets(domain_shape, idx[mask_bb], target_positions, target_radii)

        if not targets_idx.any():
            continue

        sliced = _filter_according_to_strategy(domain_shape.centroid,
                                               target_positions[targets_idx],
                                               target_radii[targets_idx],
                                               number_of_endfeet,
                                               reachout_strategy_function)
        targets_idx = targets_idx[sliced]
        domain_target_edges.extend([(domain_index, target_index) for target_index in targets_idx])

    return numpy.asarray(domain_target_edges)
