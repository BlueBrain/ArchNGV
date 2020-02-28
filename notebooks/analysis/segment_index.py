from collections import namedtuple
from functools import partial
import multiprocessing
import h5py
import numpy as np

from morphio import Morphology
import spatial_index


BATCH_SIZE = 1000


def _intersection(args):

    filepath, key, xmin, ymin, zmin, xmax, ymax, zmax = args

    with h5py.File(filepath, 'r') as fd:

        part = fd[key]

        points_dset, features_dset = part['points'], part['features']

        index = spatial_index.point_rtree(part['points'][:])
        ids = index.intersection(xmin, ymin, zmin, xmax, ymax, zmax)

        features = features_dset[:]
        features = features[ids].sum(axis=0)

        return features


class SegmentIndex:

    _features = {'length': 0, 'area': 1, 'volume': 2}

    def __init__(self, filepath):

        self._filepath = filepath

        try:
            with h5py.File(filepath, 'r') as fd:
                self._keys = list(fd.keys())
        except OSError:
            self._keys = []

        self._filepath = filepath
    
    def intersection(self, xmin, ymin, zmin, xmax, ymax, zmax, statistic='sum'):

        args = ((self._filepath, key, xmin, ymin, zmin, xmax, ymax, zmax) for key in self._keys)

        result = np.zeros(3)
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        
            for values in pool.imap_unordered(_intersection, args):
                result += values

        return result

    def append_batch(self, points, features):

        with h5py.File(self._filepath, 'a') as fd:

            n_parts = len(self._keys)
            part_name = f'part_{n_parts}'

            print(part_name, points.shape, features.shape)
        
            part = fd.create_group(part_name)

            part.create_dataset('points', data=points)
            part.create_dataset('features', data=features)

            self._keys.append(part_name)


def _construct_edges(n_points, section_lengths):

    ids = np.arange(n_points, dtype=np.intp)
    
    # [[0 1] [1 2] [2 3] [3 4] ...]
    consecutive_edges = np.repeat(ids, 2)[1:-1].reshape(-1, 2)

    # last cumsum is out of bounds (size of dataset)
    # -1 to account for N_vertices - 1 edges
    offsets = np.cumsum(section_lengths)[:-1] - 1

    return np.delete(consecutive_edges, offsets, axis=0)


def _areas(radii_starts, radii_ends, seg_lengths):
    slant_heights = np.sqrt(seg_lengths ** 2 + (radii_ends - radii_starts) ** 2)
    return np.pi * (radii_starts + radii_ends) * slant_heights


def _volumes(r_begs, r_ends, seg_lengths):
    return (1. / 3.) * np.pi * (r_begs ** 2 + r_begs * r_ends + r_ends ** 2) * seg_lengths


def _segment_features(points, edges, diameters, perimeters, is_astrocyte):
    """
    features:
        0 length
        1 area
        2 volume
    """
    radii = 0.5 * diameters
    features = np.empty((len(edges), 3), dtype=np.float32)

    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    features[:, 0] = lengths

    radii_starts, radii_ends = radii[edges.T]

    if is_astrocyte:
        if len(perimeters) > 0:
            # astrocyte area is encoded in the perimeters
            features[:, 1] = _areas(
                perimeters[edges[:, 0]] / (2. * np.pi),
                perimeters[edges[:, 1]] / (2. * np.pi),
                lengths)
        else:
            features[:, 1] = 0.0
    else:
        features[:, 1] = _areas(radii_starts, radii_ends, lengths)

    features[:, 2] = _volumes(radii_starts, radii_ends, lengths)
    return features


def _cell_segment_features(args):

    filepath, position, is_astrocyte = args

    morphology = Morphology(filepath)
    
    points = morphology.points + position

    edges = _construct_edges(
        len(points), np.diff(morphology.section_offsets))

    segment_features = _segment_features(
        points, edges,
        morphology.diameters,
        morphology.perimeters,
        is_astrocyte)

    segment_centers = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])

    return segment_centers, segment_features


def create_cell_segment_index(cell_filepaths, positions, out_filepath, is_astrocyte):

    n_cpus = multiprocessing.cpu_count()
    args = ((filepath, positions[i], is_astrocyte) for i, filepath in enumerate(cell_filepaths))

    data_index = SegmentIndex(out_filepath)

    segment_centers, segment_features = [], []

    with multiprocessing.Pool(n_cpus) as pool:

        data_generator = pool.imap_unordered(_cell_segment_features, args, chunksize=BATCH_SIZE)

        for n_cells, (points, features) in enumerate(data_generator, start=1):

            segment_centers.append(points)
            segment_features.append(features)

            if n_cells % BATCH_SIZE == 0:

                data_index.append_batch(
                    np.vstack(segment_centers),
                    np.vstack(segment_features)
                )

                segment_centers, segment_features = [], []

        if segment_centers:

            data_index.append_batch(
                np.vstack(segment_centers),
                np.vstack(segment_features)
            )

    return data_index


def create_vasculature_segment_index(vasculature, output_filepath):

    points, radii = vasculature.points, vasculature.radii

    edges = vasculature.edges
    begs, ends = points[edges.T]

    features = np.empty((len(edges), 3), dtype=np.float32)

    lengths = np.linalg.norm(ends - begs, axis=1)

    radii_starts, radii_ends = vasculature.radii[edges.T]

    features[:, 0] = lengths
    features[:, 1] = _areas(radii_starts, radii_ends, lengths)
    features[:, 2] = _volumes(radii_starts, radii_ends, lengths)

    segment_centers = 0.5 * (begs + ends)

    di = SegmentIndex(output_filepath)
    di.append_batch(segment_centers, features)

    return di
