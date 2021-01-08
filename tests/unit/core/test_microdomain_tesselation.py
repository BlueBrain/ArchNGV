import os
import pytest

import numpy as np

from archngv.core.datasets import Microdomain, MicrodomainTesselation
from archngv.building.exporters.export_microdomains import export_structure


N_CELLS = 5
MAX_NEIGHBORS = 3


class MockMicrodomainTesselation(object):

    connectivity = \
        [[j for j in range(MAX_NEIGHBORS) if i != j] for i in range(N_CELLS)]

    def __len__(self):
        return N_CELLS

    def __iter__(self):
        for i in range(N_CELLS):
            yield Microdomain(self.domain_points(i),
                              self.domain_triangle_data(i),
                              self.domain_neighbors(i))

    @property
    def flat_connectivity(self):
        return [(i, j) for i, neighbors in enumerate(self.connectivity) for j in neighbors]

    def domain_points(self, index):
        vals = np.arange(10, dtype=np.float)
        thetas = np.linspace(0., 1.8 * np.pi, 10)
        zs = np.full(10, fill_value=float(index))
        return np.column_stack((np.cos(thetas), np.sin(thetas), zs))

    def domain_triangles(self, _):
        return  np.asarray([(0, 1, 2),
                            (2, 3, 4),
                            (4, 5, 6),
                            (6, 7, 8),
                            (9, 0, 1)], dtype=np.uintp)

    def domain_triangle_data(self, _):
        polygon_ids = np.array([0, 0, 1, 1, 0], dtype=np.uintp)
        return np.column_stack((polygon_ids, self.domain_triangles(_)))

    def domain_neighbors(self, index):
        return self.connectivity[index]


@pytest.fixture(scope='session')
def directory_path(tmpdir_factory):
    return tmpdir_factory.getbasetemp()


@pytest.fixture(scope='session')
def microdomains_path(directory_path):
    return os.path.join(directory_path, 'microdomains.h5')


@pytest.fixture(scope='module')
def mockdomains(microdomains_path):

    mock_tess = MockMicrodomainTesselation()

    domains = list(iter(mock_tess))
    export_structure(microdomains_path, domains)

    return mock_tess


@pytest.fixture(scope='module')
def microdomains(microdomains_path, mockdomains):
    return MicrodomainTesselation(microdomains_path)


def test_len(microdomains, mockdomains):
    assert len(microdomains) == len(mockdomains)


def test_iter(microdomains, mockdomains):

    for mdom, fdom in zip(microdomains, mockdomains):
        assert np.allclose(mdom.points, fdom.points)
        assert np.allclose(mdom.triangles, fdom.triangles)


def test_domain_points(microdomains, mockdomains):
    for astrocyte_index in range(N_CELLS):
        assert np.allclose(microdomains.domain_points(astrocyte_index),
                           mockdomains.domain_points(astrocyte_index))


def test_domain_triangles(microdomains, mockdomains):
    for astrocyte_index in range(N_CELLS):
        assert np.allclose(microdomains.domain_triangles(astrocyte_index),
                           mockdomains.domain_triangles(astrocyte_index))


def test_domain_neighbors(microdomains, mockdomains):
    for astrocyte_index in range(N_CELLS):
        assert np.all(microdomains.domain_neighbors(astrocyte_index) == \
                      mockdomains.domain_neighbors(astrocyte_index))


def test_domain_points_object_points(microdomains):
    for i, obj in enumerate(microdomains):
        np.testing.assert_allclose(obj.points, microdomains.domain_points(i))
        np.testing.assert_allclose(obj.triangle_data, microdomains.domain_triangle_data(i))
        np.testing.assert_allclose(obj.neighbor_ids, microdomains.domain_neighbors(i, omit_walls=False))

def test_connectivity(microdomains):

    expected = np.asarray([
     [0, 1],
     [0, 2],
     [0, 3],
     [0, 4],
     [1, 2],
     [1, 3],
     [1, 4],
     [2, 3],
     [2, 4]], dtype=np.intp)

    np.testing.assert_allclose(expected, microdomains.connectivity)


def test_export_mesh(microdomains, directory_path):

    filename = os.path.join(directory_path, 'test_microdomains.stl')
    microdomains.export_mesh(filename)
