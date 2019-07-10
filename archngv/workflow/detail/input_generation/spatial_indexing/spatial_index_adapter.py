import os
import time
import rtree # TODO: unknown reference
from morphspatial import RTree # TODO: unknown reference
from morphspatial import shapes

from morphspatial.spatial_index import FastRtree  # TODO: unknown reference

import logging

L = logging.getLogger(__name__)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        L.info('method.__name__ {} seconds'.format(te - ts))

        return result

    return timed


def remove_existing_files(f):
    def wrapper(*args):

        self = args[0]

        try:

            os.remove(self.dat_filepath)
            L.info('{} already exists. Deleted.'.format(self.dat_filepath))

            os.remove(self.idx_filepath)
            L.info('{} already exists. Deleted.'.format(self.idx_filepath))

        except OSError:
            # files do not exist. Continue without issue
            pass

        return f(*args)
    return wrapper


class spatial_index(object):

    def __init__(self, filename):

        self.filename = filename

        self.dat_filepath = self.filename + '.dat'
        self.idx_filepath = self.filename + '.idx'

    @timeit
    @remove_existing_files
    def create_from_spheres(self, centers, radii):

        spheres_generator = \
        (shapes.Sphere(center, radius) for center, radius in zip(centers, radii))

        return RTree.create_from_bulk(spheres_generator, name=self.filename)

    @timeit
    @remove_existing_files
    def create_from_convex_polygons(self, shape_point_gen):

        shape_it = \
        (shapes.ConvexPolygon.from_point_cloud(pcloud) for pcloud in pclouds)

        return RTree.create_from_bulk(shape_it, name=self.filename)

    @timeit
    @remove_existing_files
    def create_from_points(self, points):

        properties = rtree.index.Property()
        properties.dimension = 3
        properties.overwrite = True
        properties.fill_factor = 0.99
        properties.leaf_capacity = resource.getpagesize()

        data_it = \
        ( (i, (x, y, z, x, y, z)) for (i, (x, y, z)) in enumerate(points) )

        return FastRtree(self.filename, data_it, properties=properties)
