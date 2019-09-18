""" NGV data exporters """

import h5py
import numpy as np


FLOAT_DTYPE = np.float32


def export_cell_placement_data(filepath, cell_ids, cell_names, somata_positions, somata_radii):
    """ Export cell data """
    cell_names = np.asarray(cell_names, dtype=bytes)

    with h5py.File(filepath, 'w') as fd:

        fd.create_dataset('ids', data=cell_ids)

        dt = h5py.special_dtype(vlen=bytes)
        fd.create_dataset('names', data=cell_names, dtype=dt)

        fd.create_dataset('positions', data=somata_positions, dtype=FLOAT_DTYPE)
        fd.create_dataset('radii', data=somata_radii, dtype=FLOAT_DTYPE)


def export_gliovascular_data(filename, endfeet_surface_coordinates, endfeet_graph_coordinates):
    """ Export endfeet gliovascular data """
    with h5py.File(filename, 'w') as fd:
        fd.create_dataset('endfoot_surface_coordinates', data=endfeet_surface_coordinates, dtype=FLOAT_DTYPE)
        fd.create_dataset('endfoot_graph_coordinates', data=endfeet_graph_coordinates, dtype=FLOAT_DTYPE)
