from .cell import Cell
import numpy as np
import os
import h5py
import logging
import numpy as np
from scipy import spatial

from .types import astrocyte_types
from .types import h5_point_map as pmap
from .types import h5_group_map as smap


L = logging.getLogger(__name__)


def _find_group(point_id, groups):
    '''Find the structure group a points id belongs to.
       Return: group or section point_id belongs to. Last group if
               point_id out of bounds.
    '''
    bs = np.searchsorted(groups[:, smap['SO']], point_id, side='right')
    bs = max(bs - 1, 0)
    return bs, groups[bs]


def _create_soma_section(radius):

    point_data = np.zeros((4, 4), dtype=np.float)

    point_data[0, pmap['xyz']] = radius * np.array((1., 0., 0.))
    point_data[1, pmap['xyz']] = radius * np.array((0., 0., 1.))
    point_data[2, pmap['xyz']] = radius * np.array((-1, 0., 0.))
    point_data[3, pmap['xyz']] = radius * np.array((0., 0., -1.))

    return point_data, np.array([[0, 1, -1]], dtype=np.intp)


class Astrocyte(Cell):

    @classmethod
    def bulk_load(cls, name, bulk_h5file):

        h5file = bulk_h5file[name]

        points = h5file['points'][:]
        groups = h5file['structure'][:]

        # fix possible section type numbering issue
        #groups[groups[:, smap['TYP']] != astrocyte_types['soma'], smap['TYP']] = astrocyte_types['process']

        try:

            perimeters = h5file['perimeters'][:]

        except KeyError:

            perimeters = np.zeros(len(points), dtype=np.float)

        return cls(name, points=points, groups=groups, perimeters=perimeters)

    @classmethod
    def load(cls, input_file, overwrite_name=None):
        """ Load astrocytes from file and assign the overwrite_name if not None,
        otherwise the name will be extracted from the filename
        """
        name = overwrite_name or os.path.basename(input_file).replace('.h5', '')

        with h5py.File(input_file, 'r+') as h5file:

            points = h5file['points'][:]
            groups = h5file['structure'][:]

            # fix possible section type numbering issue
            #groups[groups[:, smap['TYP']] != astrocyte_types['soma'], smap['TYP']] = astrocyte_types['process']

            try:

                perimeters = h5file['perimeters'][:]

            except KeyError:

                perimeters = np.zeros(len(points), dtype=np.float)

        return cls(name, points=points, groups=groups, perimeters=perimeters)

    def __init__(self, name, soma_radius=1., points=None, groups=None, perimeters=None):

        if points is None and groups is None:

            points, groups = _create_soma_section(soma_radius)

        super(Astrocyte, self).__init__(name, points, groups)

        self.perimeters = np.array([0], dtype=np.float) if perimeters is None else perimeters

        self._types = astrocyte_types

    @property
    def types(self):
        return self._types

    @property
    def endfeet(self):
        return self.sections(section_type=astrocyte_types['endfoot'])

    @property
    def processes(self):
        return self.sections(section_type=astrocyte_types['process'])

    def add_section(self, points, perimeters, section_type, section_parent):
        """ Attach a new section under section_parent.
        """
        point_offset = self.n_points
        self._points = np.vstack((self._points, points))

        group_structure = (point_offset, section_type, section_parent)

        self._groups = np.vstack((self._groups, group_structure))
        self.perimeters = np.hstack((self.perimeters, perimeters))

        return self.n_sections - 1

    def save(self, directory=None, overwrite=False, bulk_file=None):
        """ Save morphology as H5V1 format.
        """
        def write_morphology(fpointer):
            fpointer.create_dataset(name="points", data=self._points)
            fpointer.create_dataset(name="structure", data=self._groups)
            fpointer.create_dataset(name="perimeters", data=self.perimeters)

        if bulk_file is not None:

            fp = bulk_file.create_group(self.name)
            write_morphology(fp)

        else:

            file_path = os.path.join(directory, self.name + '.h5')
            mode = 'w' if overwrite else 'x'

            with h5py.File(file_path, mode) as fp:
                write_morphology(fp)
