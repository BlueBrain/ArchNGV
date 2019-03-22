

class SpatialIndexInfo(object):

    def __init__(self, ngv_config):

        self._config = ngv_config

    @property
    def synapses(self):
        raise NotImplementedError

    @property
    def vasculature(self):
        raise NotImplementedError

    @property
    def neuronal_somata(self):
        raise NotImplementedError

    @property
    def astrocytic_somata(self):
        raise NotImplementedError
