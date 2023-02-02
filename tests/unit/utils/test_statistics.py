import numpy as np

from archngv.utils.statistics import truncated_normal


def test_truncated_normal():
    mean_value = 3.1
    sdev_value = 0.001
    min_value = -1000.0
    max_value = 1000.0

    distr = truncated_normal(mean_value, sdev_value, min_value, max_value)
    values = distr.rvs(100)

    assert np.isclose(values.mean(), mean_value, atol=sdev_value)
    assert np.isclose(values.std(), sdev_value, atol=sdev_value)
    assert np.all(values > min_value)
    assert np.all(values < max_value)

    mean_value = 5.0
    sdev_value = 1.0
    min_value = 4.5
    max_value = 5.5

    distr = truncated_normal(mean_value, sdev_value, min_value, max_value)
    values = distr.rvs(100)

    assert np.all(values > min_value)
    assert np.all(values < max_value)
