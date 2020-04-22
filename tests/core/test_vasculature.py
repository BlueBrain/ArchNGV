import pathlib
import pytest
import h5py
from numpy import testing as npt
from archngv.core.datasets import Vasculature


_DATAPATH = pathlib.Path(__file__).parent.resolve() / 'data'

@pytest.fixture
def old_vasculature():
    with h5py.File(_DATAPATH / 'vasculature_old_spec.h5', 'r') as fd:

        points = fd['points'][:]
        diameters = fd['point_properties']['diameter'][:]
        edges = fd['edges'][:]

    return points, edges, diameters


@pytest.fixture
def vasculature():
    path = str(_DATAPATH / 'vasculature_new_spec.h5')
    return Vasculature.load(path)


def test_vasculature_wrapper__integration(vasculature, old_vasculature):

    old_points, old_edges, old_diameters = old_vasculature

    npt.assert_allclose(old_points, vasculature.points)
    npt.assert_array_equal(old_edges, vasculature.edges)
    npt.assert_allclose(old_diameters, vasculature.radii * 2.0)
