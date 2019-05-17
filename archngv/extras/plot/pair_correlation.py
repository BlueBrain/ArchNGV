import numpy as np
import logging
from .common import remove_spines
from .common import smooth_convolve
from archngv.extras.analysis.pair_correlation import pairCorrelationFunction_3D


L = logging.getLogger(__name__)

def plot_pair_correlation(ax, points, bounding_box):

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    x_range, y_range, z_range = bounding_box.ranges.T

    S = min(np.diff(x_range)[0],
            np.diff(y_range)[0],
            np.diff(z_range)[0])

    rMax = 0.5 * S

    dr = 1.

    L.info("dr: {}, rMax: {}".format(dr, rMax))

    g_average, radii, _ = pairCorrelationFunction_3D(x, y, z, S, rMax, dr)

    g_average = smooth_convolve(g_average, window_len=6)

    ax.plot(radii, g_average, color='k', alpha=0.9, linewidth=2.)
