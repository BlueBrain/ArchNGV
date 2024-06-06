# SPDX-License-Identifier: Apache-2.0

""" Statistics related functions """
from scipy import stats


def truncated_normal(mean_value, sdev, min_value, max_value):
    """Returns the truncated normal distribution with (mean, std)
    clipped in the range [min_value, max_value]

    Notes:
        See scipy's page for the transformation of the clipping range:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    """
    return stats.truncnorm(
        (min_value - mean_value) / sdev,
        (max_value - mean_value) / sdev,
        mean_value,
        sdev,
    )
