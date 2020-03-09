""" Synaptic Data Structures
"""

import libsonata
import numpy as np

from cached_property import cached_property

from archngv.core.common import EdgesContextManager


class SynapticData(EdgesContextManager):
    """ Synaptic data access """
    def _select(self, synapse_ids):
        if synapse_ids is None:
            return libsonata.Selection([(0, self._impl.size)])
        else:
            return libsonata.Selection(synapse_ids)

    def synapse_coordinates(self, synapse_ids=None):
        """ XYZ coordinates for given synapse_ids (all if synapse_ids not specified) """
        selection = self._select(synapse_ids)

        try:
            return np.stack([
                self._impl.get_attribute('efferent_center_x', selection),
                self._impl.get_attribute('efferent_center_y', selection),
                self._impl.get_attribute('efferent_center_z', selection),
            ]).transpose()
        except libsonata.SonataError:
            return np.stack([
                self._impl.get_attribute('afferent_center_x', selection),
                self._impl.get_attribute('afferent_center_y', selection),
                self._impl.get_attribute('afferent_center_z', selection),
            ]).transpose()

    def afferent_gids(self, synapse_ids=None):
        """ 0-based afferent neuron GIDs for given synapse_ids (all if synapse_ids not specified) """
        selection = self._select(synapse_ids)
        return self._impl.target_nodes(selection)

    @cached_property
    def n_neurons(self):
        """ Number of afferent neurons """
        return 1 + np.max(self.afferent_gids()).astype(int)  # TODO: get from HDF5 attributes

    @property
    def n_synapses(self):
        """ Number of synapses """
        return self._impl.size
