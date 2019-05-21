""" Check functions for cell placement
"""
from .exceptions import NotAlignedError


def assert_bbox_alignment(bbox1, bbox2):
    """ Checks if bounding boxes are aligned

    Arguments:
        bbox1: BoundingBox
        bbox2: BoundingBox

    Raises:
        NotAllignedError if bounding boxes are not aligned
    """
    if bbox1 != bbox2:
        msg = ('Max Point\tMin Point\n' +
               '[{:.2f} {:.2f} {:.2f}]\t'.format(*bbox1.min_point) +
               '[{:.2f} {:.2f} {:.2f}]\n'.format(*bbox2.min_point) +
               '[{:.2f} {:.2f} {:.2f}]\t'.format(*bbox1.max_point) +
               '[{:.2f} {:.2f} {:.2f}]\n'.format(*bbox2.max_point))
        raise NotAlignedError(msg)
