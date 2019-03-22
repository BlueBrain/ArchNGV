"""
Generation algorithms for spatial point pattern
"""
import logging
import numpy as np
from morphspatial import shapes

from .pattern import SpatialSpherePattern

L = logging.getLogger(__name__)


def proposal(centers, dx):
    """
    Given the centers of the voxels in the groups and the size f the voxel
    pick a random voxel and a random position in it
    """
    voxel_center = centers[np.random.randint(0, len(centers))]
    new_position = np.random.uniform(low=voxel_center - 0.5 * dx,
                                     high=voxel_center + 0.5 * dx, size=(1, 3))[0]

    return new_position


def voxels_group_centers(labels, intensity):
    """ Given the group labels of same intensity
    e.g. [0, 1, 1, 2, 0, 3, 3, 4, 4, 5 , 3, 0]
    return the grouped centers:
    [[c1, c2, c3 .. cn], [cn+1, ... cm], ...]
    which map to group indices [0, 1, 2, ..., Gn]
    """
    sorted_labels = labels.argsort()
    group_starts = np.searchsorted(labels[sorted_labels], np.unique(labels))

    # all possible indices in voxel data
    idx = np.indices(intensity.shape, dtype=np.float).reshape(3, -1).T
    centers = intensity.indices_to_positions(idx + 0.5)

    # return all the grouped voxel centers
    pairs = zip(group_starts[0: -1], group_starts[1::])

    groups = [centers[sorted_labels[i: j]] for i, j in pairs]
    groups += [centers[sorted_labels[group_starts[-1]::]]]

    return groups


class PlacementGenerator(object):
    """ The workhorse of placement
    """
    def __init__(self,
                 metropolis_hastings_parameters,
                 number_of_points,
                 cell_placement_voxel_data,
                 energy_operator,
                 other_indexes,
                 soma_radius_distribution):

        self.other_indexes = other_indexes
        self.vdata = cell_placement_voxel_data
        self.energy_operator = energy_operator
        self.soma_proposal = soma_radius_distribution

        if self.energy_operator.has_second_order_potentials():
            self.method = self._second_order
        else:
            self.method = self._first_order

        self.pattern = SpatialSpherePattern(number_of_points)

        self.__n_initial = metropolis_hastings_parameters['n_initial']
        self.__ntrials = metropolis_hastings_parameters['ntrials']
        self.__cutoff_radius = metropolis_hastings_parameters['cutoff_radius']
        self.__nmax = number_of_points
        self.__beta = metropolis_hastings_parameters['beta']

    def _is_colliding(self, xn, rn):
        """ Check if
        1. position out of bounds
        2. sphere intersects with other_indexes list
        3. sphere intersects with other spheres in the pattern
        """
        if not self.vdata.in_geometry(xn):
            return True

        if self.other_indexes:
            sphere = shapes.Sphere(xn, rn)
            for static_index in self.other_indexes:
                if static_index.is_intersecting(sphere):
                    return True

        return self.pattern.is_intersecting(xn, rn)

    def _first_order(self, centers):
        """ Sphere generation in the group of voxels with centers
        """

        dx = self.vdata.voxelized_intensity.voxel_dimensions[0]

        while 1:
            new_position = proposal(centers, dx)
            new_radius = self.soma_proposal()
            if not self._is_colliding(new_position, new_radius):
                return new_position, new_radius

    def _second_order(self, centers):
        """ Sphere generation in the group with respect to interaction
        potentials. Valid is uniformly picked in the same
        intesity group using the first order approach and an extra
        metropolis hastings optimization step is performed in order
        to minimize the energy of the potential locally for each new
        sphere
        """
        #dx = self.vdata.voxelized_intensity.voxel_dimensions[0]

        # generate some points first, say 10% of the sample
        # and return the last sphere
        if len(self.pattern) <= self.__n_initial:
            x, r = self._first_order(centers)

        # get nearest neighbor and calc its distance
        # to the current point
        index = self.pattern.nearest_neighbour(x, r)
        dist = np.linalg.norm(self.pattern.coordinates[index] - x)

        if dist > self.__cutoff_radius:
            return x, r

        # calculate the second order repulsion energy from distance
        e = self.energy_operator.second_order_potentials(dist)

        best_p = x
        best_r = r
        best_e = e

        n = 0

        # metropolis hastings procedure for minimization of the
        # repulsion energy
        while n < self.__ntrials:

            # indepedent sampler inside the voxels of the current run
            xn, rn = self._first_order(centers)

            index = self.pattern.nearest_neighbour(xn, rn)
            dist = np.linalg.norm(self.pattern.coordinates[index] - xn)

            # influence free region is a must
            if dist > self.__cutoff_radius:

                return xn, rn

            E = self.energy_operator.second_order_potentials(dist)
            logprob = self.__beta * (e - E)

            if np.log(np.random.random()) < min(0, logprob):

                x = xn
                r = rn
                e = E

            if e < best_e:

                best_e = e
                best_p = x
                best_r = r

            n += 1

        return best_p, best_r

    def run(self):
        """ Create the population of spheres
        """
        intensity = self.vdata.voxelized_intensity

        # group together voxels with identical values
        intensity_per_group, group_indices, voxels_per_group = \
            np.unique(intensity.raw, return_inverse=True, return_counts=True)

        vox_centers_per_group = voxels_group_centers(group_indices, intensity)

        counts_per_group = \
            np.round(intensity_per_group * voxels_per_group * intensity.voxel_volume * 1e-9).astype(np.intp)

        nonzero_intensity_groups = [(i, v) for (i, v) in enumerate(intensity_per_group) if not np.isclose(v, 0.0)]

        for group_index, cluster_value in nonzero_intensity_groups:

            group_total_counts = counts_per_group[group_index]
            #nuber_of_voxels = voxels_per_group[group_index]

            centers = vox_centers_per_group[group_index]

            k = 0
            while k < group_total_counts and len(self.pattern) < self.__nmax:

                new_position, new_radius = self.method(centers)
                self.pattern.add(new_position, new_radius)

                k += 1

                # some logging for iteration info
                if len(self.pattern) % int(self.__nmax * 0.1) == 0:
                    L.info('Current Number: {}'.format(len(self.pattern)))
