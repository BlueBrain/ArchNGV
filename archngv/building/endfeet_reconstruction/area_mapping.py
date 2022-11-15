""" Functions for transforming endfeet meshes from overshoot
distribution down to the biological target mesh area
"""
import numpy as np


def _empirical_cdf_probabilities(values):
    """
    Args:
        values: sorted 1D array

    Empirical distribution function

    F_n(x) = (1 / n) Sum( 1{X_i < x} )

    See: https://en.wikipedia.org/wiki/Empirical_distribution_function
    """
    n = float(len(values))
    return np.arange(1, n + 1) / n


def _sorted_nonzero_areas(areas):
    """
    Returns:
        Sorted indices of areas
        Sorted areas
    """
    sorted_idx = np.argsort(areas, kind="stable")
    sorted_areas = areas[sorted_idx]

    # don't use zero areas
    not_zero = ~np.isclose(sorted_areas, 0.0)
    sorted_idx = sorted_idx[not_zero]
    sorted_areas = sorted_areas[not_zero]

    return sorted_idx, sorted_areas


def transform_to_target_distribution(areas, target_distribution):
    """
    Args:
        areas: input array of values to transform
        biological_areas: target areas to transform to

    We want to transform random variable X (areas) to look like
    variable Y (bio areas)

    We assume that variable Y is a truncated normal. We need to scale
    down the areas so the two distributions match. Therefore we need to
    apply a transformation g: R -> R so that:

    Y = g(X) = a * X

    Because endfeet meshes can only be reduced, we take into account only
    the areas that area bigger than the target distribution. Therefore,
    the empirical cumulative function of areas will give us the probability
    that X will take value less than or equal to x, and it is this probability
    that we feed into the inverse of the bio areas to find the target area that
    we should have if the distribution was that of the biological areas.

    target areas: I(F(x))

    I = F_t(-1) : inverse of truncated normal of biological_areas: ppf
    F : empirical cumulative of areas
    x : areas

    Returns:
        Transformed area array
    """
    sorted_idx, sorted_areas = _sorted_nonzero_areas(areas)

    # empirical cdf from input areas
    pbs = _empirical_cdf_probabilities(sorted_areas)

    # use the truncated normal inverse cumulative to estimate area values
    # for the bigger areas in the cumulative
    target_areas = target_distribution.ppf(pbs)

    # transform areas that are bigger than respective bio
    mask = sorted_areas > target_areas

    transformed_areas = areas.copy()
    transformed_areas[sorted_idx[mask]] = target_areas[mask]

    return transformed_areas
