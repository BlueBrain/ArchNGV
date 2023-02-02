import numpy as np
import pytest

from archngv.building.checks import assert_bbox_alignment
from archngv.exceptions import NotAlignedError
from archngv.spatial import BoundingBox


def test_assert_bbox_alignment__aligned():
    min_point = np.array([1.0, 2.0, 3.0])
    max_point = np.array([8.0, 9.0, 10.0])

    bbox1 = BoundingBox(min_point, max_point)
    bbox2 = BoundingBox(min_point, max_point)
    bbox3 = BoundingBox(min_point, max_point)

    assert_bbox_alignment(bbox1, bbox2)
    assert_bbox_alignment(bbox2, bbox3)
    assert_bbox_alignment(bbox1, bbox3)


def test_check_alignment__tolerance():
    tolerance = 10.0

    bbox1 = BoundingBox(np.array([0.0, 0.0, 2.0]), np.array([8.0, 9.0, 10.0]))

    bbox2 = BoundingBox(np.array([1.0, 2.0, 3.0]), np.array([17.0, 9.0, 10.0]))

    bbox3 = BoundingBox(np.array([0.0, 8.0, 2.0]), np.array([8.0, 9.0, 10.0]))

    assert_bbox_alignment(bbox1, bbox3, tolerance)
    assert_bbox_alignment(bbox1, bbox2, tolerance)
    assert_bbox_alignment(bbox2, bbox3, tolerance)

    with pytest.raises(NotAlignedError):
        assert_bbox_alignment(bbox1, bbox3, tolerance=7.0)

    with pytest.raises(NotAlignedError):
        assert_bbox_alignment(bbox1, bbox2, tolerance=8.0)

    with pytest.raises(NotAlignedError):
        assert_bbox_alignment(bbox2, bbox3, tolerance=5.0)


def test_check_alignment__not_aligned():
    min_point = np.array([1.0, 2.0, 3.0])
    max_point = np.array([8.0, 9.0, 10.0])

    bbox1 = BoundingBox(min_point, max_point)
    bbox2 = BoundingBox(min_point, max_point - 13.0)
    bbox3 = BoundingBox(min_point, max_point)

    assert_bbox_alignment(bbox1, bbox3)

    with pytest.raises(NotAlignedError):
        assert_bbox_alignment(bbox1, bbox2, 10.0)

    with pytest.raises(NotAlignedError):
        assert_bbox_alignment(bbox2, bbox3, 10.0)
