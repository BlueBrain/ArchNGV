import sys # TODO: Remove
import logging # TODO: Remove
from archngv import NGVConfig # TODO: Remove
from archngv.core.data_structures.data_cells import CellData
from archngv.core.exporters.export_neuroglial_connectivity \
    import export_synapse_morphology_association # TODO: does not exist


def associate_synapses_with_morphology(cfg):

    with CellData(cfg.output_paths('cell_data')) as cell_data:
        export_synapse_morphology_association(cfg, cell_data)


