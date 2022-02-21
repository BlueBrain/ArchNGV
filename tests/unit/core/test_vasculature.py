from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import pytest
from numpy import testing as npt
from pandas import testing as pdt
from vasculatureapi import SectionVasculature

from archngv.exceptions import NGVError

DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture
def old_vasculature():
    with h5py.File(DATA_DIR / "vasculature_old_spec.h5", "r") as fd:
        points = fd["points"][:]
        diameters = fd["point_properties"]["diameter"][:]
        edges = fd["edges"][:]

    return points, edges, diameters


@pytest.fixture
def vasculature():
    return SectionVasculature.load(DATA_DIR / "vasculature_new_spec.h5").as_point_graph()


def test_vasculature_wrapper__integration(vasculature, old_vasculature):
    old_points, old_edges, old_diameters = old_vasculature

    npt.assert_allclose(old_points, vasculature.points)
    npt.assert_array_equal(old_edges, vasculature.edges)
    npt.assert_allclose(old_diameters, vasculature.diameters)
