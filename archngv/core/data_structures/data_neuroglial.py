

from .common import H5ContextManager

class NeuroglialData(H5ContextManager):
    pass

class NeuroglialDataInfo(NeuroglialData):

    def __init__(self, ngv_config):
        self._config = ngv_config
