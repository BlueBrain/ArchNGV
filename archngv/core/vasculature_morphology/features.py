""" Contains features that can be extracted from a vasculature object """

import numpy as np


def segment_lengths(vasc):
    """ Returns the distribution of segment lengths of the vasculature object
    """
    seg_starts, seg_ends = vasc.segments
    return np.linalg.norm(seg_ends - seg_starts, axis=1)


def segment_volumes(vasc):
    """ Returns the distribution of segment volumes of the vasculature object
    """
    radii_starts, radii_ends = vasc.segments_radii

    seg_lengths = segment_lengths(vasc)

    return (1. / 3.) * np.pi * (radii_starts ** 2 +
           radii_starts * radii_ends + radii_ends ** 2) * seg_lengths


def segment_slant_heights(vasc):
    """ Returns the slant heights of the truncated cone segments
    """
    radii_starts, radii_ends = vasc.segments_radii

    seg_lengths = segment_lengths(vasc)

    return np.sqrt(seg_lengths ** 2 + (radii_ends - radii_starts) ** 2)


def segment_lateral_areas(vasc):
    """ Returns the lateral areas of the trunacted cone segments """
    radii_starts, radii_ends = vasc.segments_radii

    slant_heights = segment_slant_heights(vasc)

    return np.pi * (radii_starts + radii_ends) * slant_heights


def segment_fraction_lateral_area(r_start, r_end, seg_length, point, seg_start, seg_end):
    """ Calculates the fraction surface area from the small radius
        cap to the given point inside the segment
    l : segment length
    """
    vec = point - seg_end if r_start > r_end else point - seg_start

    dR = np.abs(r_start - r_end) * np.linalg.norm(vec) / seg_length

    radius = r_start if r_start < r_end else r_end

    return np.pi * (radius + dR) * (seg_length ** 2 + dR ** 2)
