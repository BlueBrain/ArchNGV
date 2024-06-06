# SPDX-License-Identifier: Apache-2.0

"""
Energy operator functionality for calculating the potential energy of the
spatial point process
"""

import logging

import numpy as np

from archngv.building.cell_placement import potentials

POTENTIALS = {
    "spring": potentials.spring,
    "coulomb": potentials.coulomb,
    "inverse_distance": potentials.inverse_distance,
    "lenard_jones": potentials.lenard_jones,
}


L = logging.getLogger(__name__)


def _init_potentials(options):
    """Initialize potentials based on inputs from config"""
    pots = []

    for name, params in options["potentials"].items():
        try:
            # pylint: disable=cell-var-from-loop
            pots.append(lambda r: POTENTIALS[name](r, *params))
            L.info("Potential %s added with parameters %s", name, params)
        except KeyError:
            available = list(POTENTIALS.keys())
            L.warning("Key %s does not exist in potentials %s", name, available)
            raise

    return pots


class EnergyOperator:
    """Energy function class where potentials can be registered and then summed for the calculation
    of the total energy
    """

    def __init__(self, voxelized_intensity, init_options):
        self.intensity = voxelized_intensity
        self.potentials = _init_potentials(init_options)

    def has_second_order_potentials(self):
        """Checks whether there are second order interactions"""
        return len(self.potentials) > 0

    def second_order_potentials(self, pairwise_distances):
        """Second order potentials depend on the pairwise distance between objects"""
        return np.sum((func(pairwise_distances) for func in self.potentials), axis=1)

    def first_order_potentials(self, points):
        """First order potentials depend only on the position of each point"""
        return self.intensity(points)

    def __call__(self, point, distance):
        """Calculates the first and second order potentials

        Args:
            point: 1D array
                xyz
            distance: float
                distance between two points

        Returns:
            The energy based on first and second order potentials
        """
        return -self.first_order_potentials(point) + self.second_order_potentials(distance)
