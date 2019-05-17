from .common import smooth
from .common import bin_centers

from archngv.extras.analysis.histograms import laminar_histogram


def cortical_depth_plot(ax, seg_starts, seg_ends, rad_starts, rad_ends, bounding_box, n_bins,  xz_shape, measurement=None, orientation='vertical', scale=1., **kwargs):

    bins, dens, vol = laminar_histogram(seg_starts, seg_ends, rad_starts, rad_ends, bounding_box, n_bins, xz_shape=xz_shape, measurement=measurement)

    bin_cents = bin_centers(bins)

    N = dens.size * 4

    y, x = smooth(scale * dens, bin_cents, N)

    if orientation is 'horizontal':

        ax.plot(y, x, **kwargs)

    else:

        ax.plot(x, y, **kwargs)
