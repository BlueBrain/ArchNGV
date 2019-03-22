import os
import h5py
import logging
from cached_property import cached_property

from .data_cells import CellDataInfo
from .data_synaptic import SynapticDataInfo
from .data_spatial_index import SpatialIndexInfo
from .data_gliovascular import GliovascularDataInfo
from .data_microdomains import MicrodomainTesselationInfo

from ..vasculature_morphology import Vasculature

L = logging.getLogger(__name__)


class NGVData(object):

    def __init__(self, ngv_config):

        self._config = ngv_config

        self.cells    = CellDataInfo(ngv_config)

        self.synapses     = SynapticDataInfo(ngv_config)

        #self.neuroglial   = NeuroglialData(ngv_config.output_paths('NeuroglialData'))

        self.spatial_index = SpatialIndexInfo(ngv_config)
        self.gliovascular = GliovascularDataInfo(ngv_config)

        self.microdomains  = MicrodomainTesselationInfo(ngv_config)

    @cached_property
    def vasculature(self):
        return Vasculature.load(self._config.input_paths('vasculature'))

    @cached_property
    def spatial_index(self):
        return SpatialIndexInfo(self._config)

    def __enter__(self):
        self.synaptic.__enter__()
        self.cell_data.__enter__()
        #self.neuroglial.__enter__()
        self.microdomain.__enter__()
        self.gliovascular.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.synaptic.close()
        self.cell_data.close()
        #self.neuroglial.close()
        self.microdomain.close()
        self.gliovascular.close()

