""" Gliovascular connectivity exporters functions """

import h5py
import numpy as np


def write_endfoot_perspective(fd, endfeet_to_astrocyte, endfeet_to_vasculature):
    """ Write the group Endfoot and its connections to astrocyte and vasculature
    """
    endfoot_group = fd.create_group('Endfoot')

    assert len(endfeet_to_astrocyte) == len(endfeet_to_vasculature)

    connectivity_dset = \
        endfoot_group.create_dataset('connectivity', shape=(len(endfeet_to_astrocyte), 2), dtype=np.uintp)

    connectivity_dset[:, 0] = endfeet_to_astrocyte
    connectivity_dset[:, 1] = endfeet_to_vasculature

    connectivity_dset.attrs['column_names'] = \
        np.array(['astrocyte', 'vasculature_segment'], dtype=h5py.special_dtype(vlen=str))


def write_astrocyte_perspective(fd, astrocyte_to_endfeet, endfeet_to_vasculature):
    """ Write the astrocyte perspective and its connections to endfeet and vasculature segments
    """
    astrocyte_group = fd.create_group('Astrocyte')

    astrocyte_endfeet_offs = np.cumsum([0] + [len(l) for l in astrocyte_to_endfeet])
    astrocyte_endfeet_data = [endfoot for endfeet in astrocyte_to_endfeet for endfoot in endfeet]

    astrocyte_segment_data = \
        [endfeet_to_vasculature[endfoot_index] for endfeet in astrocyte_to_endfeet for endfoot_index in endfeet]

    connectivity_dset = \
        astrocyte_group.create_dataset('connectivity', shape=(len(astrocyte_endfeet_data), 2), dtype=np.uintp)

    connectivity_dset[:, 0] = astrocyte_endfeet_data
    connectivity_dset[:, 1] = astrocyte_segment_data

    offsets_dset = \
        astrocyte_group.create_dataset('offsets', data=astrocyte_endfeet_offs, dtype=np.uintp)

    connectivity_dset.attrs['column_names'] = \
        np.array(['endfoot', 'vasculature_segment'], dtype=h5py.special_dtype(vlen=str))

    offsets_dset.attrs['column_names'] = np.array(['endfoot'], dtype=h5py.special_dtype(vlen=str))


def write_vasculature_segment_perspective(fd, endfeet_to_astrocyte, endfeet_to_vasculature):
    """ Write the vasculature segment group and its connections to astrocyte and endfeet
    """
    segment_group = fd.create_group('Vasculature Segment')

    max_segment_index = np.max(endfeet_to_vasculature)
    min_segment_index = np.min(endfeet_to_vasculature)

    index_range = int(max_segment_index - min_segment_index + 1)

    segment_connectivity = \
        segment_group.create_dataset('connectivity', shape=(index_range, 2), dtype=np.intp)

    segment_connectivity[:] = -1

    remaped_idx = endfeet_to_vasculature - min_segment_index

    for n, index in enumerate(remaped_idx):

        segment_connectivity[index, 0] = n
        segment_connectivity[index, 1] = endfeet_to_astrocyte[n]

    segment_connectivity.attrs['min_index'] = min_segment_index
    segment_connectivity.attrs['max_index'] = max_segment_index

    segment_connectivity.attrs['column_names'] = \
        np.array(['endfoot', 'astrocyte'], dtype=h5py.special_dtype(vlen=str))


def create_astrocyte_to_endfeet_mapping(n_astrocytes, endfeet_to_astrocyte):
    """ For each astrocyte find the endfeet it connects to
    """
    astrocyte_to_endfeet = [[] for _ in range(n_astrocytes)]

    for endfoot_index, astro_index in enumerate(endfeet_to_astrocyte):
        astrocyte_to_endfeet[astro_index].append(endfoot_index)

    return astrocyte_to_endfeet


def export_gliovascular_connectivity(filename,
                                     n_astrocytes,
                                     endfeet_to_astrocyte,
                                     endfeet_to_vasculature):
    """ Write all the perspective into the output data file
    """
    astrocyte_to_endfeet = create_astrocyte_to_endfeet_mapping(n_astrocytes, endfeet_to_astrocyte)

    with h5py.File(filename, 'w') as fd:

        write_endfoot_perspective(fd, endfeet_to_astrocyte, endfeet_to_vasculature)
        write_astrocyte_perspective(fd, astrocyte_to_endfeet, endfeet_to_vasculature)
        write_vasculature_segment_perspective(fd, endfeet_to_astrocyte, endfeet_to_vasculature)
