import os
import pytest
from collections import namedtuple

import numpy as np

from archngv.core.data_structures.data_microdomains import MicrodomainTesselation
from archngv.core.exporters.export_microdomains import export_structure


N_CELLS = 5
MAX_NEIGHBORS = 3


MockDomain = namedtuple('MockDomain', ['points', 'triangles'])


class MockMicrodomainTesselation(object):

    connectivity = \
        [[j for j in range(MAX_NEIGHBORS) if i != j] for i in range(N_CELLS)]

    def __len__(self):
        return N_CELLS

    def __iter__(self):
        for i in range(N_CELLS):
            yield MockDomain(self.domain_points(i),
                             self.domain_triangles(i))

    @property
    def flat_connectivity():
        return [(i, j)for i, neighbors in range(self.connectivity) for j in neighbors]

    def domain_points(self, index):
        vals = np.arange(10, dtype=np.float)
        return float(index) + np.column_stack((vals, vals, vals))

    def domain_triangles(self, index):
        return [(0, 1, 2),
                (2, 3, 4),
                (4, 5, 6),
                (6, 7, 8), 
                (9, 0, 1)]

    def domain_neighbors(self, index):
        return self.connectivity[index]


@pytest.fixture(scope='session')
def microdomains_path(tmpdir_factory):

    directory_path = tmpdir_factory.getbasetemp()

    path = os.path.join(directory_path, 'microdomains.h5')

    return path


@pytest.fixture(scope='module')
def mockdomains(microdomains_path):

    domains = MockMicrodomainTesselation()
    export_structure(microdomains_path, domains)

    return domains


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

