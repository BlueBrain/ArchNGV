import numpy as np
from scipy import stats
import matplotlib.pylab as plt
from .unicode_maps import greek_alphabet_inverted
from .common import AnchoredHScaleBar

def spatial_kernel_density_plot(ax, points, x_range, y_range, cmap, levels=None, subsample_step=1):

    xmin, xmax = x_range
    ymin, ymax = y_range

    kde = stats.gaussian_kde(points[::subsample_step, :2].T)

    xx, yy = np.mgrid[xmin * 0.9: xmax * 1.1: 10, ymin * 0.9: ymax * 1.1: 100]

    density = kde(np.c_[xx.flat, yy.flat].T).reshape(xx.shape)

    if levels is not None:

        cset = ax.contourf(xx, yy, density, cmap=cmap, levels=levels)

    else:

        cset = ax.contourf(xx, yy, density, cmap=cmap)

    mu = greek_alphabet_inverted['mu']

    label = '200 {}m'.format(mu)
    ob = AnchoredHScaleBar(ax=ax, size=200, extent=0.01, label=label, loc=4, frameon=False,
                           pad=0.6, sep=4, color="k") 

    ax.add_artist(ob)

    return cset.levels
