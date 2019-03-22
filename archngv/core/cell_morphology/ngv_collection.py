import os
import h5py
import logging
import numpy as np
import pandas as pd

try:
    from voxcell import CellCollection
except ImportError:
    from voxcell.core import CellCollection

from .astrocyte import Astrocyte
from .types import h5_point_map as pmap
from .types import h5_group_map as smap

import collections

MicrodomainCollection = collections.namedtuple('MicrodomainCollection', 'structure points triangles')
GliovascularConnectivity = collections.namedtuple('GliovascularConnectivity', 'structure graph_points surface_points')

L = logging.getLogger(__name__)


def radius_h5(path, estimation_function):
    with h5py.File(path, 'r') as f:
        try:

            points = f['points'][:f['structure'][1, smap['SO']], pmap['xyz']]

        except ValueError:

            points = f['points'][:, pmap['xyz']]

        center = np.mean(points, axis=0)
        radius = estimation_function(np.linalg.norm(points - center, axis=1))
    return radius


def morphology_abspaths(morph_names, morph_dir):
    return np.array([os.path.join(morph_dir, m) + '.h5' for m in morph_names])


class AstrocyteCollection(CellCollection):

    @classmethod
    def load(cls, ngv_config):
        c_path = ngv_config.circuit_path
        return cls(ngv_config, collection=CellCollection.load(c_path))

    def __init__(self):
        super(AstrocyteCollection, self).__init__()
        self.diameters = None
        self._microdomains = MicrodomainCollection()
        self._gliovascular_connectivity = GliovascularConnectivity()

    @property
    def properties(self):
        return self.properties

    def add_gliovascular_connectivity(self, graph_targets, surface_targets):
        """ gv conn
        """
        grph_targets = {key: targets for key, targets in graph_targets.groupby(lambda key: key[0])}
        surf_targets = {key: targets for key, targets in surface_targets.groupby(lambda key: key[0])}

    def add_microdomains(self, microdomains):
        """ Add microdomains
        """
        points = []
        triangles = []
        structure = []

        point_offset = 0
        triangle_offset = 0

        for domain in microdomains:

            structure.append((point_offset, triangle_offset))

            ps = domain.points
            ts = domain.triangles

            points.extend(ps.tolist())
            triangles.extend(ts.tolist())

            point_offset += len(ps)
            triangle_offset += len(ts)

        self.microdomains.points = np.asarray(points, dtype=np.float)
        self.microdomains.triangles = np.asarray(triangles, dtype=np.int)
        self.microdomains.structure = np.asarray(structure, dtype=np.uintp)

    def __add__(self, collection):
        raise NotImplementedError
        new_collection = CellCollection()
        new_collection.positions = np.vstack((coll1.positions, coll2.positions))
        new_collection.orientations = np.vstack((coll1.orientations, coll2.orientations))
        new_collection.properties = pd.concat([coll1.properties, coll2.properties])
        return new_collection

    def _mask_by_property(self, property_type, property_value):
        return (self.properties[property_type] == property_value).values

    def morphologies_by_mtype(self, mtype):

        mask = self._mask_by_property('mtype', mtype)

        names = self.properties['names'][mask]

        return morphology_abspaths(names, self._config.morphology_directory)

    def mtype_positions(self, mtype):
        mask = self._mask_by_property('mtype', mtype)
        return self.positions[mask]

    @property
    def astrocyte_paths(self):
        return self.morphologies_by_mtype('ASTROCYTE')

    def radii_from_morphologies(self, directory, mask=None, estimation_function=np.mean):

        fpaths = morphology_abspaths(self.properties['names'], directory)
        pos = self.positions
        if mask is not None:

            fpaths = fpaths[mask]
            pos = pos[mask]
        return np.fromiter((radius_h5(path, estimation_function) for path in fpaths), dtype=np.float)

    """
    def create_empty_morphologies(self, positions, orientations, radii, properties, cell_class):
        ''' Cerates a set of empty morphologies
        '''

        self.add_properties(pd.DataFrame(properties))

        self.positions = positions

        self.orientations = orientations

        cells = [cell_class(name=name, soma_radius=radius) for name, radius in  zip(properties['names'], radii)]

        map(lambda i: cells[i].rotate(self.orientations[i]), range(len(cells)))
        map(lambda i: cells[i].translate(self.positions[i]), range(len(cells)))

        map(lambda cell: cell.save(self._config.morphology_directory, overwrite=True), cells)

        return morphology_abspaths(properties['names'], self._config.morphology_directory)

    def create_morphologies_from_raw(self, positions, orientations, radii, properties, cell_class, raw_dir):

        morph_paths = \
            [os.path.join(raw_dir, name) for name in filter(lambda x: x.endswith('.h5'), os.listdir(raw_dir))]
        n_cells = positions.shape[0]

        morph_paths = np.random.choice(morph_paths, n_cells, replace=True)

        self.add_properties(properties)

        self.positions = positions
        self.orientations = orientations

        for i, morph_path in enumerate(morph_paths):

            cell = cell_class.load(morph_path, overwrite_name=properties['names'][i])

            cell.rotate(self.orientations[i])

            cell.translate(self.positions[i])

            cell.save(self._config.morphology_directory, overwrite=True)

        return morphology_abspaths(properties['names'], self._config.morphology_directory)
    """

    def save(self, directory):
        self._collection.save(directory)
