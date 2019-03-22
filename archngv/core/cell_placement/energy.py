"""
Energy operator functionality for calculating the potential energy of the
spatial point process
"""

import logging
import numpy as np

from . import potentials


POTENTIALS = {'spring': potentials.spring,
              'coulomb': potentials.coulomb,
              'inverse_distance': potentials.inverse_distance,
              'lenard_jones': potentials.lenard_jones}


L = logging.getLogger(__name__)


def _init_potentials(options):
    """ Initialize potentials based on inputs from config
    """
    pots = []

    for name, params in options['potentials'].items():

        try:

            pots.append(lambda r: POTENTIALS[name](r, *params))

            L.info('Potential {} added with parameters {}'.format(name, params))

        except KeyError:

            available = list(POTENTIALS.keys())
            L.warning('Key {} does not exist in potentials {}'.format(name, available))
            raise

    return pots


class EnergyOperator(object):
    """ Energy function class where potentials can be registered and then summed for the calculation
    of the total energy
    """
    def __init__(self, intensity, init_options):

        self.intensity = intensity
        self.potentials = _init_potentials(init_options)

    def has_second_order_potentials(self):
        """ Checks whether there are second order interactions
        """
        return len(self.potentials) > 0

    def second_order_potentials(self, pairwise_distances):
        """ Second order potentials depend on the pairwise distance between objects
        """
        return np.sum((func(pairwise_distances) for func in self.potentials), axis=1)

    def first_order_potentials(self, points):
        """ First order potentials depend only on the position of each point
        """
        return self.intensity(points)

    def __call__(self, point, distance):
        return - self.first_order_potentials(point) \
               + self.second_order_potentials(distance)
