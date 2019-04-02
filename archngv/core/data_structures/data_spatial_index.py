""" Spatial index info
"""

class SpatialIndexInfo(object):
    """ Access to out of core spatial indexes
    """
    def __init__(self, ngv_config):

        self._config = ngv_config

    @property
    def synapses(self):
        """ Synapses spatial index """
        raise NotImplementedError

    @property
    def vasculature(self):
        """ Vasculature spatial index """
        raise NotImplementedError

    @property
    def neuronal_somata(self):
        """ Neuronal somata spatial index """
        raise NotImplementedError

    @property
    def astrocytic_somata(self):
        """ Astrocytic somata spatial index """
        raise NotImplementedError
