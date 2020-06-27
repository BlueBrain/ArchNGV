import pathlib
import pytest
import h5py
from numpy import testing as npt
from pandas import testing as pdt

from archngv.core.datasets import Vasculature
from tempfile import NamedTemporaryFile


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


def test_vasculature_sonata_cycle(vasculature):

    with NamedTemporaryFile(suffix='.h5') as tfile:

        filename = tfile.name

        # load regular morphology file into PointVasculature
        v1 = vasculature

        # write node population
        v1.save_sonata(filename)

        # load sonata node population file into PointVasculature
        v2 = Vasculature.load_sonata(filename)

        # check that they are identical
        pdt.assert_frame_equal(v1.node_properties, v2.node_properties)
        pdt.assert_frame_equal(v1.edge_properties, v2.edge_properties)

