import numpy as np


def azimuthal_average(hist_data, bin_edges, grid_centers,center=None, dr=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).

    """


    n_dims = len(bin_edges)

    rmax = 0.5 * np.abs(bin_edges[0][-1] - bin_edges[0][0])

    # unravel dd hisgrogram
    data_flat = hist_data.ravel()

    if center is None:
        center = np.array([bin_edges[n][0] + 0.5 * (bin_edges[n][-1] - bin_edges[n][0]) for n in range(n_dims)])

    grid_radii = np.sqrt(sum((grid_centers[n].ravel() - center[n]) ** 2 for n in range(n_dims)))

    if dr is None:

        dr = bin_edges[0][1] - bin_edges[0][0]

    r = 0.

    radial_profile = []
    radial_bins = [r]

    while r < rmax:

        mask = (r <= grid_radii) & (grid_radii < r + dr)

        if mask.any():

            radial_profile.append(data_flat[mask].mean())

        else:

            radial_profile.append(0.)

        r += dr

        radial_bins.append(r)

    return np.asarray(radial_bins), np.asarray(radial_profile)


def spatial_fourier_transform(points):

    ft = np.fft.fftn(points)
    return np.fft.fftshift(ft)


def spatial_fourier_magnitude(ft):
    return np.abs(ft)


def spatial_fourier_phase(ft):
    return np.angle(ft)


def spatial_periodogram(points):

    ft = spatial_fourier_transform(points)
    return spatial_fourier_magnitude(ft)


def histogram_bin_centers(bin_edges):

    centers_by_dim = []

    for i, dim in enumerate(bin_edges):

        # find the offset to the center of the bin
        dc = 0.5 * (dim[1] - dim[0])

        # find the centroid for each bin by adding the offset
        centers_by_dim.append(dim[0:-1] + dc)

    return np.meshgrid(*centers_by_dim)


def fourier_power_spectrum(h):

    p = spatial_periodogram(h)

    return np.power(p, 2)


def power_spectrum_radial_average(points):

    p = _spatial_periodogram(points)

    return azimuthal_average(p)

