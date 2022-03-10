import os

import numpy as np
from numpy import testing as npt
import pytest

from archngv.building.exporters.export_microdomains import export_structure
from archngv.core.datasets import Microdomain, MicrodomainTesselation

N_CELLS = 5
MAX_NEIGHBORS = 3


class MockMicrodomainTesselation(object):

    connectivity = [[j for j in range(MAX_NEIGHBORS) if i != j] for i in range(N_CELLS)]

    def __len__(self):
        return N_CELLS

    def __iter__(self):
        for i in range(N_CELLS):
            yield Microdomain(
                self.domain_points(i), self.domain_triangle_data(i), self.domain_neighbors(i)
            )

    @property
    def flat_connectivity(self):
        return [(i, j) for i, neighbors in enumerate(self.connectivity) for j in neighbors]

    def domain_points(self, index):
        vals = np.arange(10, dtype=np.float32)
        thetas = np.linspace(0.0, 1.8 * np.pi, 10)
        zs = np.full(10, fill_value=float(index))
        return np.column_stack((np.cos(thetas), np.sin(thetas), zs))

    def domain_triangles(self, _):
        return np.asarray([(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (1, 3, 5)], dtype=np.uintp)

    def domain_triangle_data(self, _):
        polygon_ids = np.array([0, 0, 1, 1, 0], dtype=np.uintp)
        return np.column_stack((polygon_ids, self.domain_triangles(_)))

    def domain_neighbors(self, index):
        return self.connectivity[index]


@pytest.fixture(scope="session")
def directory_path(tmpdir_factory):
    return tmpdir_factory.getbasetemp()


@pytest.fixture(scope="session")
def microdomains_path(directory_path):
    return os.path.join(directory_path, "microdomains.h5")


@pytest.fixture(scope="module")
def mockdomains(microdomains_path):

    mock_tess = MockMicrodomainTesselation()

    domains = list(iter(mock_tess))
    export_structure(microdomains_path, domains)

    return mock_tess


@pytest.fixture(scope="module")
def microdomains(microdomains_path, mockdomains):
    return MicrodomainTesselation(microdomains_path)


def test_len(microdomains, mockdomains):
    assert len(microdomains) == len(mockdomains)


def test_iter(microdomains, mockdomains):

    for mdom, fdom in zip(microdomains, mockdomains):
        npt.assert_allclose(mdom.points, fdom.points)
        npt.assert_allclose(mdom.triangles, fdom.triangles)


def test_domain_points(microdomains, mockdomains):
    for astrocyte_index in range(N_CELLS):
        npt.assert_allclose(
            microdomains.domain_points(astrocyte_index), mockdomains.domain_points(astrocyte_index)
        )


def test_domain_triangles(microdomains, mockdomains):
    for astrocyte_index in range(N_CELLS):
        npt.assert_allclose(
            microdomains.domain_triangles(astrocyte_index),
            mockdomains.domain_triangles(astrocyte_index),
        )


def test_domain_neighbors(microdomains, mockdomains):
    for astrocyte_index in range(N_CELLS):
        npt.assert_array_equal(
            microdomains.domain_neighbors(astrocyte_index),
            mockdomains.domain_neighbors(astrocyte_index),
        )


def test_domain_objects(microdomains):

    scaling_factors = np.linspace(0.1, 100.0, len(microdomains))

    for domain_index, domain in enumerate(microdomains):

        npt.assert_allclose(domain.points, microdomains.domain_points(domain_index))
        npt.assert_allclose(
            domain.triangle_data,
            microdomains.domain_triangle_data(domain_index),
        )
        npt.assert_allclose(
            domain.neighbor_ids, microdomains.domain_neighbors(domain_index, omit_walls=False)
        )

        scale_factor = scaling_factors[domain_index]
        new_domain = domain.scale(scale_factor)

        npt.assert_allclose(
            new_domain.points,
            scale_factor * (domain.points - domain.centroid) + domain.centroid,
        )


def test_connectivity(microdomains):

    expected = np.asarray(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4]], dtype=np.int32
    )

    npt.assert_allclose(expected, microdomains.connectivity)


def test_export_mesh(microdomains, directory_path):

    filename = os.path.join(directory_path, "test_microdomains.stl")
    microdomains.export_mesh(filename)
