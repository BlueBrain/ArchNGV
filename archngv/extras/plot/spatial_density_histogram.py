import numpy as np
from .common import smooth
from .common import bin_centers
from .common import plot_oriented
from ...analysis.histograms import mass_histogram


def plot_laminar_mass_density(axis, positions, bounding_box, options, plt_options):

    bins, densities = mass_histogram(positions, bounding_box, options)

    scale = options['scale']

    bin_cents = bin_centers(bins)

    N = densities.size * 4

    y, x = smooth(scale * densities, bin_cents, N)

    plot_oriented(axis, x, y, options['orientation'], **plt_options)


def plot_laminar_endeet_contact_mass_density(axis, circuits, options, plt_options):

    scale = options['scale']

    data = []

    for i, circuit in enumerate(circuits):

        ngv_data = circuit.data

        targets = ngv_data.gliovascular.endfoot_surface_coordinates

        bounding_box = ngv_data.vasculature.bounding_box

        bins, densities = mass_histogram(targets, bounding_box, options)


        bin_cents = bin_centers(bins)

        axis.plot(bin_cents, scale * densities, color='gray', linewidth=0.5)

        data.append(densities)

    mean_densities = np.mean(data, axis=0)

    bin_cents = bin_centers(bins)

    N = mean_densities.size * 3

    #y_mean, x_mean = smooth(scale * mean_densities, bin_cents, N)

    axis.plot(bin_cents, scale * mean_densities, color='k', linewidth=2)

    if len(circuits) > 1:

        sdev_densities = np.std(data, axis=0)
        #y_sdev, x_sdev = smooth(scale * sdev_densities, bin_cents, N)
        axis.fill_between(bin_cents,
                          scale*mean_densities - scale*sdev_densities,
                          scale*mean_densities + scale*sdev_densities, color='gray', alpha=0.6)

    #axis.set_ylim([np.min(scale * mean_densities) * 0.99, np.max(scale * mean_densities)])


