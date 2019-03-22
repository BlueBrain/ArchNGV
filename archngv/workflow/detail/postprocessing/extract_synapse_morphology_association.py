import sys
import logging
from archngv import NGVConfig
from archngv.data_structures.ngv_data import CellData
from archngv.exporters.export_neuroglial_connectivity import export_synapse_morphology_association


def associate_synapses_with_morphology(cfg):

    with CellData(cfg.output_paths('cell_data')) as cell_data:
        export_synapse_morphology_association(cfg, cell_data)


