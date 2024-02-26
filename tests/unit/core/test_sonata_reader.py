from unittest.mock import patch

import libsonata
import numpy.testing as npt
import pytest
from utils import get_data

import archngv.core.sonata_readers as tested
from archngv.exceptions import NGVError


@patch("libsonata.NodeStorage.population_names", return_value=["A", "B"])
def test_bad_population(mock_libsonata):
    with pytest.raises(NGVError):
        tested.NodesReader(get_data("nodes.h5"))


class TestNodesReader:
    def setup_method(self):
        self.nodes = tested.NodesReader(get_data("nodes.h5"))

    def test_name(self):
        assert self.nodes.name == "default"

    def test_property_names(self):
        assert self.nodes.property_names == {
            "z",
            "mtype",
            "y",
            "model_type",
            "x",
            "morphology",
            "layer",
            "model_template",
        }

    def test_get_properties(self):
        npt.assert_equal(self.nodes.get_properties("layer"), [[1], [2], [3]])
        npt.assert_equal(self.nodes.get_properties(["layer"]), [[1], [2], [3]])
        npt.assert_equal(self.nodes.get_properties("layer", ids=0), [[1]])
        npt.assert_equal(self.nodes.get_properties("layer", ids=[0]), [[1]])
        npt.assert_equal(self.nodes.get_properties("layer", ids=libsonata.Selection([0])), [[1]])

        npt.assert_equal(
            self.nodes.get_properties(["model_template", "morphology"]),
            [
                ["hoc:small_bio", "morph-A"],
                ["hoc:small_bio", "morph-B"],
                ["hoc:small_bio", "morph-C"],
            ],
        )

        with pytest.raises(NGVError):
            self.nodes.get_properties(["unknown"])

        with pytest.raises(NGVError):
            self.nodes.get_properties("unknown")

        with pytest.raises(NGVError):
            self.nodes.get_properties(["rotation_angle_zaxis", "unknown"])

        # 2 different types int and float
        with pytest.raises(NGVError):
            self.nodes.get_properties(["layer", "x"])

    def test_len(self):
        assert len(self.nodes) == 3


class TestEdgesReader:
    def setup_method(self):
        self.edges = tested.EdgesReader(get_data("edges.h5"))

    def test_name(self):
        assert self.edges.name == "default"

    def test_property_names(self):
        assert self.edges.property_names == {
            "afferent_center_x",
            "afferent_center_y",
            "afferent_center_z",
            "efferent_section_pos",
            "syn_weight",
        }

    def test_get_properties(self):
        npt.assert_equal(self.edges.get_properties("syn_weight"), [[1.0], [1.0], [1.0], [1.0]])
        npt.assert_equal(self.edges.get_properties(["syn_weight"]), [[1.0], [1.0], [1.0], [1.0]])
        npt.assert_equal(self.edges.get_properties("syn_weight", ids=0), [[1.0]])
        npt.assert_equal(self.edges.get_properties("syn_weight", ids=[0]), [[1.0]])
        npt.assert_equal(
            self.edges.get_properties("syn_weight", ids=libsonata.Selection([0])), [[1.0]]
        )

        npt.assert_equal(
            self.edges.get_properties(["syn_weight", "efferent_section_pos"]),
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        )
        npt.assert_equal(
            self.edges.get_properties(["efferent_section_pos", "syn_weight"]),
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        )

        with pytest.raises(NGVError):
            self.edges.get_properties(["unknown"])

        with pytest.raises(NGVError):
            self.edges.get_properties("unknown")

        with pytest.raises(NGVError):
            self.edges.get_properties(["efferent_section_pos", "unknown"])

        # 2 different types int and float
        with pytest.raises(NGVError):
            self.edges.get_properties(["efferent_section_pos", "efferent_section_id"])

    def test_len(self):
        assert len(self.edges) == 4

    def test_afferent_edges(self):
        npt.assert_equal(self.edges.afferent_edges(0), [0])
        npt.assert_equal(self.edges.afferent_edges(1), [1, 2, 3])
        npt.assert_equal(self.edges.afferent_edges(2), [])

        npt.assert_equal(self.edges.afferent_edges([0, 1]), [0, 1, 2, 3])
        npt.assert_equal(self.edges.afferent_edges([1, 0]), [0, 1, 2, 3])

    def test_efferent_edges(self):
        npt.assert_equal(self.edges.efferent_edges(0), [1, 2])
        npt.assert_equal(self.edges.efferent_edges(1), [])
        npt.assert_equal(self.edges.efferent_edges(2), [0, 3])

        npt.assert_equal(self.edges.efferent_edges([2, 0]), [0, 1, 2, 3])
        npt.assert_equal(self.edges.efferent_edges([0, 2]), [0, 1, 2, 3])

    def test_afferent_nodes(self):
        npt.assert_equal(self.edges.afferent_nodes(0), [2])
        npt.assert_equal(self.edges.afferent_nodes(1), [0, 2])
        npt.assert_equal(self.edges.afferent_nodes(2), [])

        npt.assert_equal(self.edges.afferent_nodes([0, 1]), [0, 2])

    def test_efferent_nodes(self):
        npt.assert_equal(self.edges.efferent_nodes(0), [1])
        npt.assert_equal(self.edges.efferent_nodes(1), [])
        npt.assert_equal(self.edges.efferent_nodes(2), [0, 1])

        npt.assert_equal(self.edges.efferent_nodes([0, 2]), [0, 1])
