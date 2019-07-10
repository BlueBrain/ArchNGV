""" Check functions for cell placement
"""
import logging
import numpy as np
from archngv.core.exceptions import NotAlignedError


L = logging.getLogger(__name__)


def assert_bbox_alignment(bbox1, bbox2, tolerance=10.0):
    """ Checks if bounding boxes are aligned

    Arguments:
        bbox1: BoundingBox
        bbox2: BoundingBox
        tolerance: float
            Alignment tolerance in micrometers

    Raises:
        NotAllignedError if bounding boxes are not aligned
    """
    def message(bbox1, bbox2):
        return ('\n' +
               'Min points: [{:.2f} {:.2f} {:.2f}]\t'.format(*bbox1.min_point) +
               '[{:.2f} {:.2f} {:.2f}]\n'.format(*bbox2.min_point) +
               'Max Points: [{:.2f} {:.2f} {:.2f}]\t'.format(*bbox1.max_point) +
               '[{:.2f} {:.2f} {:.2f}]\n'.format(*bbox2.max_point))

    if not bbox1 == bbox2:

        max_aligned = np.allclose(bbox1.max_point, bbox2.max_point, atol=tolerance)
        min_aligned = np.allclose(bbox1.min_point, bbox2.min_point, atol=tolerance)

        msg = message(bbox1, bbox2)

        if max_aligned and min_aligned:
            L.warning('Datasets aligned within tolerance %f um\n%s', tolerance, msg)
        else:
            raise NotAlignedError(msg)
