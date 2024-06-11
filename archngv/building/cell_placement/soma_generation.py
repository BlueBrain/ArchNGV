# SPDX-License-Identifier: Apache-2.0

"""
Functions related to the soma generation
"""
from scipy import stats


def truncated_normal_distribution(soma_data):
    """Returns spipy.stats distribution for soma radii
    truncated normal in the interval [a, b]
    soma_data: (mean, std, a, b)
    """
    soma_mean, soma_sdev, low, high = soma_data
    return stats.truncnorm(
        low - soma_mean / soma_sdev,
        (high - soma_mean) / soma_sdev,
        loc=soma_mean,
        scale=soma_sdev,
    ).rvs
