""" In memory container of microdomain tesselation
"""

import logging

L = logging.getLogger(__name__)

class MicrodomainTesselation(object):
    """ Microdomain tesselation Data structure. It stores the regions (convex triangular polygon)
    and the connectivity between the regions in order to find nearest neighbors.
    """
    #@classmethod
    #def load(cls, filepath, order_by=None):
    #    regions, connectivity = load_structure(filepath, order_by)
    #    return cls(regions, connectivity)

    def __init__(self, regions, connectivity):

        self._regions = regions
        self._connectivity = connectivity

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, item):
        return self.regions[item]

    def __iter__(self):
        return iter(self.regions)

    @property
    def regions(self):
        """ Returns regions """
        return self._regions

    @property
    def connectivity(self):
        """ Connectivity between microdomains
        """
        return self._connectivity

    def with_regions(self, regions):
        """ Returns a tesselation that shares the connectivity
        but has new regions
        """
        return MicrodomainTesselation(regions, self.connectivity)
