import logging
import numpy as np

log = logging.getLogger(__name__)


def check_validity(func):

    def wrapper(*args, **kwargs):

        assert args[0].shape == args[1].shape, (args[0].shape, args[1].shape)

        return func(*args, **kwargs)

    return wrapper


def create_laminar_bins(bounding_box, n_bins):

    _, (ymin, ymax), _ = bounding_box.ranges.T

    return np.linspace(ymin, ymax, n_bins)


def _cylinder_area(p0, p1, r0, r1):

    seg_len = np.linalg.norm(p1 - p0)

    slt_hght = np.sqrt(seg_len ** 2 + (r1 - r0) ** 2)

    return np.pi * (r0 + r1) * slt_hght


def _cylinder_length(starts, ends):
    return np.linalg.norm(ends - starts, axis=1)


def vectors_included_in_bin_1D(vec_starts, vec_ends, bin_start, bin_end):
    return ((bin_start <= vec_starts[:, 1]) & (vec_ends[:, 1] <= bin_end)) | \
           ((bin_start <= vec_ends[:, 1]) & (vec_starts[:, 1] <= bin_end))


def points_included_in_bin_1D(ps, bin_start, bin_end):
    return (bin_start <= ps[:, 1]) & (ps[:, 1] <= bin_end)


def calc_bin_volume(dy, x_extent, z_extent, xz_shape):

    if xz_shape is 'rectangle':

        v =  dy * x_extent * z_extent

    elif xz_shape is 'circle':

        v =  dy * 2. * np.pi * x_extent * z_extent

    else:

        raise TypeError

    log.debug('bin_volume: {} um^3'.format(v))
    return v


_MEASUREMENTS = {'length': lambda starts, ends, r_starts, r_ends: _cylinder_length(starts, ends),
                 'area'  : lambda starts, ends, r_starts, r_ends: _cylinder_area(starts, ends, r_starts, r_ends)}

@check_validity
def laminar_histogram(vec_starts,
                      vec_ends,
                      r_starts,
                      r_ends,
                      bounding_box,
                      n_bins,
                      xz_shape='rectangle',
                      measurement='length'):

    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounding_box.ranges

    bins = np.linspace(ymin, ymax, n_bins)

    density = np.zeros(bins.size - 1, dtype=np.float)

    bin_volume = calc_bin_volume(bins[1] - bins[0],
                                 xmax - xmin,
                                 zmax - zmin,
                                 xz_shape)

    for i in range(n_bins - 1):

        mask_inside = vectors_included_in_bin_1D(vec_starts, vec_ends, bins[i], bins[i + 1])

        if mask_inside.any():
            density[i] = _MEASUREMENTS[measurement](vec_starts[mask_inside],
                                                    vec_ends[mask_inside],
                                                    r_starts[mask_inside],
                                                    r_ends[mask_inside]).sum()

    density /= bin_volume

    return bins, density, bin_volume

def get_measurement_function(options, bins, bounding_box):

    name = options['measurement_function']

    if name is 'mean':

        m_func = np.mean

    elif name is 'density':

        xz_shape = options['xz_shape']

        ex, _, ez = bounding_box.extent

        bin_volume = calc_bin_volume(bins[1] - bins[0],
                                     ex,
                                     ez,
                                     xz_shape)

        m_func = lambda vals: np.sum(vals) / bin_volume

    elif name is 'total':

        m_func = np.sum

    elif name is 'count':

        m_func = len

    else:

        raise TypeError 

    return m_func


def mass_histogram(points, bounding_box, options):

    laminar_bins = create_laminar_bins(bounding_box, options['n_bins'])

    measurements = np.zeros(laminar_bins.size - 1, dtype=np.float)

    measurement_function = get_measurement_function(options, laminar_bins, bounding_box)

    for i in range(laminar_bins.size - 1):

        mask_inside = points_included_in_bin_1D(points, laminar_bins[i], laminar_bins[i + 1])

        if mask_inside.any():

            measurements[i] = measurement_function(mask_inside)

    return laminar_bins, measurements


def wiring_histogram(vec_starts, vec_ends, bounding_box, options):


    laminar_bins = create_laminar_bins(bounding_box, options['n_bins'])

    measurements = np.zeros(laminar_bins.size - 1, dtype=np.float)

    measurement_function = get_measurement_function(options, laminar_bins, bounding_box)
    for i in range(laminar_bins.size - 1):

        mask_inside = vectors_included_in_bin_1D(vec_starts, vec_ends, laminar_bins[i], laminar_bins[i + 1])
        if mask_inside.any():

            lengths = np.linalg.norm(vec_ends[mask_inside] - vec_starts[mask_inside], axis=1)

            measurements[i] = measurement_function(lengths)

    return laminar_bins, measurements
