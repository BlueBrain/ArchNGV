""" Container for NGV connectome """
import logging
import numpy as np
from archngv.core.common import EdgesContextManager, H5ContextManager


L = logging.getLogger(__name__)


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


class AstrocyteEntry:
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
        return slice(self._offsets[astrocyte_index],
                     self._offsets[astrocyte_index + 1])

    def to_endfoot(self, astrocyte_index):
        """ Endfeet indices for astrocyte """
        slice_ = self._offset_slice(astrocyte_index, self._offset_t['endfoot'])
        return self._connectivity[slice_]


class EndfootEntry:
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
            'vasculature_section_id': 1,
            'vasculature_segment_id': 2
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
        """ Vasculature section and segment id for endfoot """

        cols = [self._target_t['vasculature_section_id'],
                self._target_t['vasculature_segment_id']]

        return self._connectivity[endfoot_index, cols]


class NeuroglialConnectivity(EdgesContextManager):
    """ Neuroglial connectivity access """

    def _synapse_selection(self, astrocyte_id):
        return self._impl.efferent_edges(astrocyte_id)

    def astrocyte_synapses(self, astrocyte_id):
        """ Synapse IDs corresponding to a given `astrocyte_id` """
        selection = self._synapse_selection(astrocyte_id)
        return self._impl.get_attribute('synapse_id', selection)

    def astrocyte_neurons(self, astrocyte_id):
        """ post-synaptic neurons given an `astrocyte_id` """
        selection = self._synapse_selection(astrocyte_id)
        return np.unique(self._impl.target_nodes(selection))


class GlialglialConnectivity(EdgesContextManager):
    """ Glialglial connectivity access
    """
    def astrocyte_astrocytes(self, astrocyte_id):
        """ Astrocyte connected to astrocyte with `astrocyte_id` """
        selection = self._impl.efferent_edges(astrocyte_id)
        return self._impl.target_nodes(selection)
