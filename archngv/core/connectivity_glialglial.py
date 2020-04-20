""" Data structures for connectivity between astrocytes
"""
from archngv.core.common import EdgesContextManager


class GlialglialConnectivity(EdgesContextManager):
    """ Glialglial connectivity access
    """
    def astrocyte_astrocytes(self, astrocyte_id):
        """ Astrocyte connected to astrocyte with `astrocyte_id` """
        selection = self._impl.efferent_edges(astrocyte_id)
        return self._impl.target_nodes(selection)
