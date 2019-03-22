import os
import logging

import numpy as np

from .common import H5ContextManager


L = logging.getLogger(__name__)


class GliovascularConnectivity(H5ContextManager):

    def __init__(self, filepath):
        super(GliovascularConnectivity, self).__init__(filepath)

        self.endfoot = EndfootEntry(self._fd)
        self.astrocyte = AstrocyteEntry(self._fd)
        self.vasculature_segment = VasculatureSegmentEntry(self._fd)

    @property
    def n_astrocytes(self):
        return len(self.astrocyte._offsets) - 1

    @property
    def n_endfeet(self):
        return len(self.endfoot._connectivity)

    @property
    def edges_astrocyte_endfeet(self):
        e2a = self.endfoot.to_astrocyte_map
        endfeet_indices = np.arange(len(e2a), dtype=np.uintp)
        return np.column_stack((e2a, endfeet_indices))


class AstrocyteEntry(object):

    def __init__(self, fd):

        self._target_t = \
        {
            'endfoot': 0,
            'vasculature_segment': 1
        }

        self._offset_t = \
        {
            'endfoot': 0
        }

        self._connectivity = fd['/Astrocyte/connectivity']
        self._offsets = fd['/Astrocyte/offsets']

    def _offset_slice(self, astrocyte_index, offset_type):
        # right now the array is 1d because there is only
        # one offset
        return self._offsets[astrocyte_index], \
               self._offsets[astrocyte_index + 1]

    def to_endfoot(self, astrocyte_index):
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['endfoot'])
        return self._connectivity[beg: end, self._target_t['endfoot']]

    def to_vasculature_segment(self, astrocyte_index):
        beg, end = self._offset_slice(astrocyte_index, self._offset_t['endfoot'])
        return self._connectivity[beg: end, self._target_t['vasculature_segment']]


class EndfootEntry(object):

    def __init__(self, fd):

        self._target_t = \
        {
            'astrocyte': 0,
            'vasculature_segment': 1
        }

        self._connectivity = fd['/Endfoot/connectivity']

    def to_astrocyte(self, endfoot_index):
        return self._connectivity[endfoot_index, self._target_t['astrocyte']]

    @property
    def to_astrocyte_map(self):
        return self._connectivity[:, self._target_t['astrocyte']]

    def to_vasculature_segment(self, endfoot_index):
        return self._connectivity[endfoot_index, self._target_t['vasculature_segment']]


class VasculatureSegmentEntry(object):

    def __init__(self, fd):

        self._target_t = \
        {
            'endfoot': 0,
            'astrocyte': 1
        }

        vasculature_group = fd['/Vasculature Segment']
        self._connectivity = vasculature_group['connectivity']

        self._min_index = self._connectivity.attrs['min_index']
        self._max_index = self._connectivity.attrs['max_index']

    def _is_index_valid(self, segment_index):
        return self._min_index <= segment_index <= self._max_index

    def to_endfoot(self, segment_index):
        return self._connectivity[segment_index - self._min_index, \
                                  self._target_t['endfoot']] if self._is_index_valid(segment_index) else None

    def to_astrocyte(self, segment_index):
        return self._connectivity[segment_index - self._min_index, \
                                  self._target_t['astrocyte']] if self._is_index_valid(segment_index) else None


