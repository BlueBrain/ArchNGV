""" Data structures for connectivity between neurons and astrocytes
"""

from archngv.core.common import EdgesContextManager


POPULATION_NAME = 'neuroglial'


class NeuroglialConnectivity(EdgesContextManager):
    """ "Neuroglial connectivity access """
    def astrocyte_synapses(self, astrocyte_id):
        """ Synapse IDs corresponding to a given `astrocyte_id` """
        selection = self._impl.efferent_edges(astrocyte_id)
        return self._impl.get_attribute('synapse_id', selection)

    def astrocyte_neurons(self, astrocyte_id):
        """ post-synaptic neurons given an `astrocyte_id` """
        selection = self._impl.efferent_edges(astrocyte_id)
        return self._impl.target_nodes(selection)
