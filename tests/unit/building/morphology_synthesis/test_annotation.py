from collections import namedtuple

import numpy as np
import pytest
from numpy import testing as npt

from archngv.building.morphology_synthesis import annotation as tested

MockSection = namedtuple("MockSection", ["points", "id"])


class MockCell:
    def __init__(self, sections):
        self.sections = sections

    def iter(self):
        return self.sections


@pytest.fixture
def cell():
    sections = [
        MockSection(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0],
                    [0.0, 0.3, 0.0],
                    [0.0, 0.4, 0.0],
                ]
            ),
            0,
        ),
        MockSection(
            np.array([[1.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.4, 0.0, 0.0], [1.5, 0.0, 0.0]]), 1
        ),
    ]

    return MockCell(sections)


def test_morphology_unwrapped(cell):
    df = tested._morphology_unwrapped(cell)

    npt.assert_allclose(df.x, [0.0, 0.0, 0.0, 1.1, 1.3, 1.45])
    npt.assert_allclose(df.y, [0.1, 0.25, 0.35, 0.0, 0.0, 0.0])
    npt.assert_allclose(df.z, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    npt.assert_array_equal([0, 0, 0, 1, 1, 1], df.section_id)
    npt.assert_array_equal([0, 1, 2, 0, 1, 2], df.segment_id)
    npt.assert_allclose([0.1, 0.05, 0.05, 0.1, 0.1, 0.05], df.segment_offset)
    npt.assert_allclose([0.25, 0.625, 0.875, 0.2, 0.6, 0.9], df.section_position)
