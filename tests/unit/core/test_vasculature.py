from pathlib import Path
import h5py

import pytest
from numpy import testing as npt
from pandas import testing as pdt

from archngv.core.datasets import Vasculature
from archngv.exceptions import NGVError

from tempfile import NamedTemporaryFile


DATA_DIR = Path(__file__).resolve().parent / 'data'


@pytest.fixture
def old_vasculature():
    with h5py.File(DATA_DIR / 'vasculature_old_spec.h5', 'r') as fd:
        points = fd['points'][:]
        diameters = fd['point_properties']['diameter'][:]
        edges = fd['edges'][:]

    return points, edges, diameters


@pytest.fixture
def vasculature():
    return Vasculature.load(DATA_DIR / 'vasculature_new_spec.h5')


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
        pdt.assert_frame_equal(v1.node_properties, v2.node_properties, check_dtype=False, check_index_type=False)

        pdt.assert_frame_equal(v1.edge_properties, v2.edge_properties, check_dtype=False, check_index_type=False)


class TestVasculature:
    def setup(self):
        self.vasculature = Vasculature.load(DATA_DIR / 'vasculature_new_spec.h5')

    def test_fail_init(self):
        with pytest.raises(NGVError):
            Vasculature([])

    def test_load_sonata(self):
        Vasculature.load_sonata(DATA_DIR / 'vasculature_sonata.h5')

    def test_node_properties(self):
        assert list(self.vasculature.node_properties) == ["x", "y", "z", "diameter"]
        npt.assert_allclose(
            self.vasculature.node_properties.loc[:2, ["x", "y", "z", "diameter"]].to_numpy(),
            [[0., 0., 4650., 20.],
             [0., 0., 4665., 22.309322],
             [0., 0., 4680., 23.771172]])

    def test_edge_properties(self):
        assert list(self.vasculature.edge_properties) == ['start_node', 'end_node', 'type']
        assert self.vasculature.edge_properties.loc[(0, 0)].tolist() == [0, 1, 0]

    def test_points(self):
        npt.assert_allclose(self.vasculature.points[:3], [[0., 0., 4650.],
                                                          [0., 0., 4665.],
                                                          [0., 0., 4680.]])

    def test_edges(self):
        npt.assert_allclose(self.vasculature.edges[:3], [[0, 1], [1, 2], [2, 3]])

    def test_radii(self):
        npt.assert_allclose(self.vasculature.radii[:3], [10., 11.154661, 11.885586])

    def test_segment_radii(self):
        # TODO: need to check this I am not sure this is the correct data orientation
        npt.assert_allclose(self.vasculature.segment_radii[:, 0], [10., 11.154661])

    def test_segment_points(self):
        # TODO: need to check this I am not sure this is the correct data orientation
        npt.assert_allclose(self.vasculature.segment_points[:, 0], [[0., 0., 4650.],
                                                                    [0., 0., 4665.]])

    def test_bounding_box(self):
        from archngv.spatial import BoundingBox
        assert isinstance(self.vasculature.bounding_box, BoundingBox)

    def test_volume(self):
        # TODO : need to find a better test
        npt.assert_allclose([self.vasculature.volume], [48488240.0])
        assert self.vasculature.volume == 48488240.0

    def test_area(self):
        # TODO : need to find a better test
        npt.assert_allclose([self.vasculature.area], [9535104.0])

    def test_length(self):
        # TODO : need to find a better test
        npt.assert_allclose([self.vasculature.length], [151500.34])

    def test_point_graph(self):
        from vasculatureapi.utils.adjacency import AdjacencyMatrix

        res = self.vasculature.point_graph
        assert isinstance(res, AdjacencyMatrix)
        assert res.n_vertices == len(self.vasculature.node_properties)

    def test_map_edges_to_sections(self):
        assert self.vasculature.map_edges_to_sections[0] == \
               self.vasculature.edge_properties.index[0][0]
