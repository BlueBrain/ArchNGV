import morphio
import numpy as np
import pytest
from numpy import testing as npt

from archngv.building.morphology_synthesis import full_astrocyte as tested
from archngv.exceptions import NGVError


def test_post_growing():
    m = morphio.mut.Morphology()
    m.soma.points = np.array([[1.0, 1.0, 1.0]])
    m.soma.diameters = np.array([0.0])

    m.append_root_section(
        morphio.PointLevel(np.array([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]]), np.array([1.0, 2.0])),
        morphio.SectionType.basal_dendrite,
    )
    m.append_root_section(
        morphio.PointLevel(np.array([[1.8, 1.9, 1.0], [2.4, 2.5, 2.6]]), np.array([1.0, 2.0])),
        morphio.SectionType.axon,
    )

    res = tested._post_growing(m, np.array([1.0, 1.0, 1.0]))

    # check NEURON ordering
    npt.assert_array_equal(
        [int(s.type) for s in res.root_sections],
        [2, 3],
    )

    # check translation back to origin
    npt.assert_allclose(res.soma.points, [[0.0, 0.0, 0.0]])
    npt.assert_allclose(
        np.vstack([s.points for s in res.iter()]),
        [
            [0.8, 0.9, 0.0],
            [1.4, 1.5, 1.6],
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        atol=1e-6,
    )


def test_sanity_check__missing_duplicate_points():
    m = morphio.mut.Morphology()
    m.soma.points = np.array([[1.0, 1.0, 1.0]])
    m.soma.diameters = np.array([0.0])

    root = m.append_root_section(
        morphio.PointLevel(np.array([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]]), np.array([1.0, 2.0])),
        morphio.SectionType.basal_dendrite,
    )

    root.append_section(
        morphio.PointLevel(np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]), np.array([1.0, 2.0])),
        morphio.SectionType.basal_dendrite,
    )

    with pytest.raises(NGVError):
        tested._sanity_checks(m)


def test_sanity_check__one_point_sections():
    m = morphio.mut.Morphology()
    m.soma.points = np.array([[1.0, 1.0, 1.0]])
    m.soma.diameters = np.array([0.0])

    root = m.append_root_section(
        morphio.PointLevel(np.array([[1.1, 1.2, 1.3]]), np.array([1.0])),
        morphio.SectionType.basal_dendrite,
    )

    with pytest.raises(NGVError):
        tested._sanity_checks(m)


def test_sanity_check__unifurcations():
    m = morphio.mut.Morphology()
    m.soma.points = np.array([[1.0, 1.0, 1.0]])
    m.soma.diameters = np.array([0.0])

    root = m.append_root_section(
        morphio.PointLevel(np.array([[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]]), np.array([1.0, 2.0])),
        morphio.SectionType.basal_dendrite,
    )

    root.append_section(
        morphio.PointLevel(np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]), np.array([1.0, 2.0])),
        morphio.SectionType.basal_dendrite,
    )

    with pytest.raises(NGVError):
        tested._sanity_checks(m)
