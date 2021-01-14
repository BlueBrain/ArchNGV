import morphio
import numpy as np
from numpy import testing as npt
import pytest
from unittest.mock import Mock

from archngv.building.morphology_synthesis import endfoot_compartment as tested
from archngv.core.datasets import EndfootMesh


def _rot2D(p, theta):

    c = np.cos(theta)
    s = np.sin(theta)

    p1 = p.copy()

    if p.ndim == 1:
        p1[0] = p[0] * c - p[1] * s
        p1[1] = p[0] * s + p[1] * c
    else:
        p1[:, 0] = p[:, 0] * c - p[:, 1] * s
        p1[:, 1] = p[:, 0] * s + p[:, 1] * c

    return p1


def test_extent_across_vasculature_segment_medial_axis():
    """
      0,1,0
        |  . (0.25, 0.75, 0.0)
        |
        |    * ref_point (1.0, 0.0, 0.0)
        |
        |  . (0.25, -0.75, 0.0)
      0,-1,0
    """

    ref_point = np.array([1.0, 0.0, 0.0])

    segment = np.array([
        [0.,1.,0.],
        [0.,-1.,0.]
    ])

    points = np.array([
        [0.25, 0.75, 0.0],
        [0.25, -0.75, 0.0]
    ])

    npt.assert_allclose(
        tested._extent_across_vasculature_segment_medial_axis(
            points, ref_point, segment),
        0.75 + 0.75
    )

    npt.assert_allclose(
        tested._extent_across_vasculature_segment_medial_axis(
            points, ref_point, segment[[1, 0]]),
        0.75 + 0.75
    )

    # adding the symmetric points should not change the result
    points = np.array([
        [0.25, 0.75, 0.0],
        [0.25, -0.75, 0.0],
        [-0.25, 0.75, 0.0],
        [-0.25, -0.75, 0.0]
    ])

    npt.assert_allclose(
        tested._extent_across_vasculature_segment_medial_axis(
            points, ref_point, segment),
        0.75 + 0.75
    )

    # rotations should not change the result
    theta = np.pi / 4.
    ref_point = _rot2D(ref_point, theta)
    points = _rot2D(points, theta)
    segment = _rot2D(segment, theta)

    npt.assert_allclose(
        tested._extent_across_vasculature_segment_medial_axis(
            points, ref_point, segment),
        0.75 + 0.75
    )

    theta = np.pi / 4.
    ref_point = _rot2D(ref_point, theta)
    points = _rot2D(points, theta)
    segment = _rot2D(segment, theta)

    npt.assert_allclose(
        tested._extent_across_vasculature_segment_medial_axis(
            points, ref_point, segment),
        0.75 + 0.75
    )


def test_endfoot_compartment_features():

    area = 3.12
    thickness = 1.1
    length = 1.32

    expected_volume = area * thickness

    diameter, perimeter = tested._endfoot_compartment_features(length, area, thickness)

    volume = np.pi  * (0.5 * diameter) ** 2 * length

    npt.assert_allclose(volume, expected_volume)
    npt.assert_allclose(length * perimeter, area)


def test_create_endfeet_compartment_data():

    segments = np.array([
        [
            [0., 0., 0.],
            [0., 1., 0.]
        ],
        [
            [0., 0., 1.],
            [0., 0., 0.]
        ],
        [
            [1., 0., 0.],
            [0., 0., 0.]
        ],
        [
            [1., 0., 0.],
            [0., 0., 0.]
        ]
    ])
    
    ref_points = np.array([
        [1.0, 0.5, 0.0],
        [0.0, 1.0, 0.5],
        [0.5, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ])

    meshes = [
        EndfootMesh(
            index=0,
            points=np.array([
                [0.25, 0.75, 0.0],
                [0.25, 0.25, 0.0]
            ]),
            triangles=np.array([[0, 1, 0]]),
            area=4.0, thickness=0.5 * np.pi
        ),
        EndfootMesh(
            index=1,
            points=np.array([
                [0.0, 0.25, 0.75],
                [0.0, 0.25, 0.25]
            ]),
            triangles=np.empty(shape=(0, 3), dtype=np.int),
            area=4.0, thickness=0.5 * np.pi
        ),
        EndfootMesh(
            index=2,
            points=np.array([
                [0.75, 0.0, 0.25],
                [0.25, 0.0, 0.25]
            ]),
            triangles=np.array([[0, 1, 0]]),
            area=4.0, thickness=0.5 * np.pi
        ),
        EndfootMesh(
            index=2,
            points=np.array([
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ]),
            triangles=np.array([[0, 1, 0]]),
            area=4.0, thickness=0.5 * np.pi
        )
    ]

    lengths, diameters, perimeters = tested.create_endfeet_compartment_data(
        segments, ref_points, meshes)

    # 2nd has no triangles and 4th zero length
    npt.assert_allclose(lengths, [0.5, 0., 0.5, 0.])
    npt.assert_allclose(diameters, [4., 0., 4., 0.])
    npt.assert_allclose(perimeters, [8., 0., 8., 0.])
