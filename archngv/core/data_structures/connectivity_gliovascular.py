""" Classes for accessing the connectivity between astrocytes
and the vasculature
"""
import numpy as np

from archngv.core.data_structures.common import H5ContextManager


class GliovascularConnectivity(H5ContextManager):
    """
    Arguments:
        filepath:
            Absilute path to the hdf5 file.
    Attributes:
        endfoot:
            Endfoot view allows accesing connectivity
            from the endfoot to astrocyte and vasculature_segment
        astrocyte:
            Astrocyte view allows accessing connectivity
            from the astrocyte to the endfoot and vasculature_segment.
        vasculature_segment:
            Vasculature segment view allows accesing connectivity
            from the astrocyte to the endfoot and astrocyte.
    """
    def __init__(self, filepath):
        super(GliovascularConnectivity, self).__init__(filepath)

        self.endfoot = EndfootEntry(self._fd)
        self.astrocyte = AstrocyteEntry(self._fd)
        self.vasculature_segment = VasculatureSegmentEntry(self._fd)

    @property
    def n_astrocytes(self):
        """ Number of astrocytes """
        return len(self.astrocyte)

    @property
    def n_endfeet(self):
        """ Number of endfeet """
        return len(self.endfoot)

    @property
    def edges_astrocyte_endfeet(self):
        """
        Returns: array[int, (N, 2)]
            Each row determines a connectivity edge (i-astro, j-endfoot)
        """
        e2a = self.endfoot.to_astrocyte_map
        endfeet_indices = np.arange(len(e2a), dtype=np.uintp)
        return np.column_stack((e2a, endfeet_indices))


class AstrocyteEntry(object):
    """ Astrocytic point of view. Allows access to all its
    neighbors.

    Attributes:
        connectivity: hdf5 Dataset[int, (N, 2)]
            Column 1: Endfeet ids
            Column 2: Vasculature Segment ids
        offsets : hdf5 Dataset[init, (M + 1, 1)]
            The connectivity corresponding to the i-th
            astrocyte can be accesed as
            connectivity[offsets[i]: offsets[i + 1]].
            Note that for M astrocytes there are M + 1 rows
            as the end offest of the last astrocyte is contained
            as well. This is different than the usual h5v1 spec where
            it is left to the user to extract the last section from the
            number of points.
    """
    def __init__(self, fd):

        self._target_t = {
            'endfoot': 0,
            'vasculature_segment': 1
        }

        self._offset_t = {
            'endfoot': 0
        }

        self._connectivity = fd['/Astrocyte/connectivity']
        self._offsets = fd['/Astrocyte/offsets']

    def __len__(self):
        """ Size """
        return len(self._offsets) - 1

    def _offset_slice(self, astrocyte_index, _):
        # right now the array is 1d because there is only
        # one offset
        return self._offsets[astrocyte_index], \
               self._offsets[astrocyte_index + 1]

    def to_endfoot(self, astrocyte_index):
        """ Endfeet indices for astrocyte """
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['endfoot'])
        return self._connectivity[beg: end, self._target_t['endfoot']]

    def to_vasculature_segment(self, astrocyte_index):
        """ Vasculature indices for astrocyte """
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['endfoot'])
        return self._connectivity[beg: end, self._target_t['vasculature_segment']]


class EndfootEntry(object):
    """ Endfoot point of view. Allows access to all its
    neighbors.

    Attributes:
        connectivity: hdf5 Dataset[int, (N, 2)]
            Column 1: Astrocyte ids
            Column 2: Vasculature Segment ids
            N is the number of endfeet.
    """
    def __init__(self, fd):

        self._target_t = {
            'astrocyte': 0,
            'vasculature_segment': 1
        }

        self._connectivity = fd['/Endfoot/connectivity']

    def __len__(self):
        return len(self._connectivity)

    def to_astrocyte(self, endfoot_index):
        """ Astrocyte index for endfoot """
        return self._connectivity[endfoot_index, self._target_t['astrocyte']]

    @property
    def to_astrocyte_map(self):
        """ Astrocyte connectivity """
        return self._connectivity[:, self._target_t['astrocyte']]

    def to_vasculature_segment(self, endfoot_index):
        """ Vasculature segment index for endfoot """
        return self._connectivity[endfoot_index, self._target_t['vasculature_segment']]


class VasculatureSegmentEntry(object):
    """ Vasculature point of view. Allows access to all its
    neighbors.

    Attributes:
        connectivity: hdf5 Dataset[int, (N, 2)]
            Column 1: Endfeet ids
            Column 2: Astrocyte ids
            N is the number of segments in the interval [min_index, max_index]
            where min_index, max_index are the smallest and biggest ids with a
            connection. In between it is expected to have segments without connections.
        min_index: int
            Smallest segment id with a connection.
        max_index: int
            Biggest segment id with a connection.
    """
    def __init__(self, fd):

        self._target_t = {
            'endfoot': 0,
            'astrocyte': 1
        }

        vasculature_group = fd['/Vasculature Segment']
        self._connectivity = vasculature_group['connectivity']

        self._min_index = self._connectivity.attrs['min_index']
        self._max_index = self._connectivity.attrs['max_index']

    def _is_index_valid(self, segment_index):
        """ Check if the given index is inside the interval [min_index, max_index]
        outside of which there are not connections.
        """
        return self._min_index <= segment_index <= self._max_index

    def to_endfoot(self, segment_index):
        """ Endfoot index """
        translated_index = segment_index - self._min_index
        if self._is_index_valid(segment_index):
            return self._connectivity[translated_index, self._target_t['endfoot']]
        return None

    def to_astrocyte(self, segment_index):
        """ Astrocyte index """
        translated_index = segment_index - self._min_index
        if self._is_index_valid(segment_index):
            return self._connectivity[translated_index, self._target_t['astrocyte']]
        return None
