import os
import h5py
import pytest

import numpy
"""
from ..gliovascular_core import GVData


_path = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_path, '../../../../test_data')


@pytest.fixture
def gv_data():

    gd = GVData()

    idx = numpy.arange(4)

    gt = gd.graph_targeting
    gt.positions = idx[:, numpy.newaxis] * numpy.ones(3)
    gt.vasculature_segment = idx
    gt.astrocyte_target_edges = numpy.array([[0, 3], [1, 2]])
    reversed_idx = idx[5: 0: -1]

    st = gd.surface_targeting
    st.positions = reversed_idx[:, numpy.newaxis] * numpy.ones(3)
    st.vasculature_segment = reversed_idx
    st.astrocyte_target_edges = numpy.array([[1, 2],[3, 4]])

    return gd


def test_io_gv_data(gv_data):

    filename = os.path.join(DATA_PATH, 'test_write_gv_data.h5')

    gv_data.save(filename)

    ld_data = GVData.load(filename)

    assert numpy.allclose(gv_data.graph_targeting.positions,
                          ld_data.graph_targeting.positions)

    assert numpy.allclose(gv_data.graph_targeting.vasculature_segment,
                          ld_data.graph_targeting.vasculature_segment)

    assert numpy.allclose(gv_data.graph_targeting.astrocyte_target_edges,
                          ld_data.graph_targeting.astrocyte_target_edges)

    assert numpy.allclose(gv_data.surface_targeting.positions,
                          ld_data.surface_targeting.positions)

    assert numpy.allclose(gv_data.surface_targeting.vasculature_segment,
                          ld_data.surface_targeting.vasculature_segment)

    assert numpy.allclose(gv_data.surface_targeting.astrocyte_target_edges,
                          ld_data.surface_targeting.astrocyte_target_edges)

"""
