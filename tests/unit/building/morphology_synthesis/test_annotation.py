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

    points, df_locations = tested._morphology_unwrapped(cell)

    npt.assert_allclose(
        points,
        [
            [0.0, 0.10, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.35, 0.0],
            [1.1, 0.0, 0.0],
            [1.3, 0.0, 0.0],
            [1.45, 0.0, 0.0],
        ],
    )

    npt.assert_array_equal([0, 0, 0, 1, 1, 1], df_locations.section_id)
    npt.assert_array_equal([0, 1, 2, 0, 1, 2], df_locations.segment_id)
    npt.assert_allclose([0.1, 0.05, 0.05, 0.1, 0.1, 0.05], df_locations.segment_offset)
    npt.assert_allclose([0.25, 0.625, 0.875, 0.2, 0.6, 0.9], df_locations.section_position)
