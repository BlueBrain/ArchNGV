import pytest
from morphspatial import BoundingBox

import numpy as np

from ..checks import assert_bbox_alignment
from ..exceptions import NotAlignedError


def test_assert_bbox_alignment__aligned():

    min_point = np.array([1., 2., 3.])
    max_point = np.array([8., 9., 10.])

    bbox1 = BoundingBox(min_point, max_point)
    bbox2 = BoundingBox(min_point, max_point)
    bbox3 = BoundingBox(min_point, max_point)

    assert_bbox_alignment(bbox1, bbox2)
    assert_bbox_alignment(bbox2, bbox3)
    assert_bbox_alignment(bbox1, bbox3)


def test_check_alignment__not_aligned():

    min_point = np.array([1., 2., 3.])
    max_point = np.array([8., 9., 10.])

    bbox1 = BoundingBox(min_point, max_point)
    bbox2 = BoundingBox(min_point, max_point - 1.0)
    bbox3 = BoundingBox(min_point, max_point)

    assert_bbox_alignment(bbox1, bbox3)

    with pytest.raises(NotAlignedError):
        assert_bbox_alignment(bbox1, bbox2)

    with pytest.raises(NotAlignedError):
        assert_bbox_alignment(bbox2, bbox3)

