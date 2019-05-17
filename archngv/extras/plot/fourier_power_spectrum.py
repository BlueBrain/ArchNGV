import numpy as np
import pylab as plt

from archngv.extras.analysis.fourier_transform import azimuthal_average
from archngv.extras.analysis.fourier_transform import histogram_bin_centers
from archngv.extras.analysis.fourier_transform import fourier_power_spectrum

def plot_fourier_power_spectrum(points, normalize_intensity=True):

    n_dim = points.shape[1]

    h, bin_edges = np.histogramdd(points, bins=30)

    # normalize intensity to remove the DC term from the FFT
    if normalize_intensity:
        h -= h.ravel().mean()

    bin_centers = histogram_bin_centers(bin_edges)

    power_spectrum = fourier_power_spectrum(h)

    if n_dim == 2:

        f1, ax1 = plt.subplots(1, 1, figsize=(10,10))
        ax1.imshow(power_spectrum, cmap='gist_gray', interpolation='None')
        ax1.set_title('Data Points')

    else:

        f2, ax2 = plt.subplots(10, 1, figsize=(10, 20))

        s = h.shape[0]

        if s % 2 == 0:

            i = s / 2

            idx = list(range(i - 5, i + 5))

        else:

            i = (s + 1) / 2

            idx = list(range(i - 4, i)) + [i] + list(range(i + 1, i + 5))

        for i, ax in enumerate(ax2):

            ax.imshow(h[..., i], cmap='gist_gray', interpolation='None')

            ax.set_axis_off()

    f2, ax2 = plt.subplots(1, 1, figsize=(10,10))
    ax2.scatter(points[:, 0], points[:, 1], c='k', s=0.01)

    radial_bins, prof = azimuthal_average(h, bin_edges, bin_centers)

    f3, ax3 = plt.subplots(1, 1, figsize=(10,10))

    ax3.plot(radial_bins[1:], prof)

    ax3.set_title('Mean Radial Power')
    ax3.set_xlabel('Radial Distance')
    ax3.set_ylabel('Mean Power')

    plt.show()
