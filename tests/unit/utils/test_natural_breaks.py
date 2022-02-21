import numpy as np

from archngv.utils.natural_breaks import kmeans_1d


def test_kmeans_1d_1():
    values = np.array([1.1, 1.2, 1.3, 10.1, 10.2, 10.3, 100.1, 100.2, 100.3])

    results = kmeans_1d(values, 1)
    np.testing.assert_allclose([0], results)

    results = kmeans_1d(values, 2)
    np.testing.assert_allclose([0, 6], results)

    results = kmeans_1d(values, 3)
    np.testing.assert_allclose([0, 3, 6], results)

    results = kmeans_1d(values, 4)
    np.testing.assert_allclose([0, 3, 6, 7], results)


def test_kmeans_1d_2():

    values = np.array([1.0, 10.0, 100.0])

    results = kmeans_1d(values, 3)
    np.testing.assert_allclose([0, 1, 2], results)

    values = np.array([1.1, 1.2, 10.0, 100.0])

    results = kmeans_1d(values, 3)
    np.testing.assert_allclose([0, 2, 3], results)

    results = kmeans_1d(values, 2)
    np.testing.assert_allclose([0, 3], results)

    results = kmeans_1d(values, 1)
    np.testing.assert_allclose([0], results)

    values = np.array([1.1, 1.2, 10.0, 10.1, 100.0])

    results = kmeans_1d(values, 3)
    np.testing.assert_allclose([0, 2, 4], results)

    results = kmeans_1d(values, 2)
    np.testing.assert_allclose([0, 4], results)

    results = kmeans_1d(values, 1)
    np.testing.assert_allclose([0], results)


def test_kmeans_1d_3():
    values = np.array([1.1, 1.2, 1.3, 10.1, 10.1, 10.1, 100.2, 100.2, 100.3])

    results = kmeans_1d(values, 3)
    np.testing.assert_allclose([0, 3, 6], results)

    values = np.array([1.1, 1.1, 1.1, 1.1])
    results = kmeans_1d(values, 3)
    np.testing.assert_allclose([0, 2, 3], results)
