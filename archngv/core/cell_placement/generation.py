"""
Generation algorithms for spatial point pattern
"""
# pylint: disable = too-many-locals, too-many-arguments, too-many-instance-attributes

import logging
from collections import namedtuple

import numpy as np
from morphspatial import shapes

from .pattern import SpatialSpherePattern


L = logging.getLogger(__name__)


def proposal(voxel_centers, voxel_edge_length):
    """
    Given the centers of the voxels in the groups and the size f the voxel
    pick a random voxel and a random position in it.

    Args:
        voxel_centers: 2D array
            Coordinates of the centers of vocels.
        voxel_edge_length: float
            Edge length od voxel

    Returns: 1D array
        Coordinates of uniformly chosen voxel center
    """
    random_index = np.random.randint(0, len(voxel_centers))
    voxel_center = voxel_centers[random_index]

    new_position = np.random.uniform(low=voxel_center - 0.5 * voxel_edge_length,
                                     high=voxel_center + 0.5 * voxel_edge_length, size=(1, 3))[0]

    return new_position


def voxels_group_centers(labels, intensity):
    """ Given the group labels of same intensity
    e.g. [0, 1, 1, 2, 0, 3, 3, 4, 4, 5 , 3, 0]
    return the grouped centers:
    [[c1, c2, c3 .. cn], [cn+1, ... cm], ...]
    which map to group indices [0, 1, 2, ..., Gn]

    Args:
        labels: array[int]
            Labels corresponding to group for each voxel
        intensity: PlacementVoxelData
            Voxel data containing intensities

    Returns: list of lists
        Voxel centers per group
    """
    sorted_labels = labels.argsort()
    group_starts = np.searchsorted(labels[sorted_labels], np.unique(labels))

    # all possible indices in voxel data
    idx = np.indices(intensity.shape, dtype=np.float).reshape(3, -1).T
    voxel_centers = intensity.indices_to_positions(idx + 0.5)

    # return all the grouped voxel centers
    pairs = zip(group_starts[0: -1], group_starts[1::])

    groups = [voxel_centers[sorted_labels[i: j]] for i, j in pairs]
    groups += [voxel_centers[sorted_labels[group_starts[-1]::]]]

    return groups


PlacementParameters = namedtuple('PlacementParameters', ['beta',
                                                         'number_of_trials',
                                                         'cutoff_radius',
                                                         'initial_sample_size'])

class PlacementGenerator:
    """ The workhorse of placement

    Args:
        parameters:

        total_spheres: int
            The number of spheres that will be generated.
        voxel_data: PlacementVoxelData
            Atlas voxelized intensity and regions.
        energy_operator: EnergyOperator
            Function object that calculates the potential for a new
            placement operation.
        index_list: list[rtree]
            List of static spatial indexes, i.e. the indexes that
            are not changed during the simulation.
        soma_radius_distribution:
            Soma radius sampler

    Attrs:
        pattern:
            The empty collection for the spheres that we will place
            in space.
        method:
            The energy method to be used.

    """
    def __init__(self, parameters, total_spheres, voxel_data,
                 energy_operator, index_list, soma_radius_distribution):

        self.vdata = voxel_data
        self.index_list = index_list
        self.parameters = parameters
        self.energy_operator = energy_operator
        self.soma_proposal = soma_radius_distribution

        if self.energy_operator.has_second_order_potentials():
            self.method = self.second_order
        else:
            self.method = self.first_order

        self.pattern = SpatialSpherePattern(total_spheres)
        self._total_spheres = total_spheres

    def is_colliding(self, trial_position, trial_radius):
        """ Check if
        1. position out of bounds
        2. sphere intersects with other_indexes list
        3. sphere intersects with other spheres in the pattern
        """
        if not self.vdata.in_geometry(trial_position):
            return True

        if self.index_list:
            sphere = shapes.Sphere(trial_position, trial_radius)
            for static_index in self.index_list:
                if static_index.is_intersecting(sphere):
                    return True

        return self.pattern.is_intersecting(trial_position, trial_radius)

    def first_order(self, voxel_centers):
        """ Sphere generation in the group of voxels with centers
        """
        voxel_edge_length = \
            self.vdata.voxelized_intensity.voxel_dimensions[0]

        while 1:

            new_position = proposal(voxel_centers, voxel_edge_length)
            new_radius = self.soma_proposal()

            if not self.is_colliding(new_position, new_radius):
                return new_position, new_radius

    def second_order(self, voxel_centers):
        """ Sphere generation in the group with respect to interaction
        potentials. Valid is uniformly picked in the same
        intesity group using the first order approach and an extra
        metropolis hastings optimization step is performed in order
        to minimize the energy of the potential locally for each new
        sphere
        """
        # dx = self.vdata.voxelized_intensity.voxel_dimensions[0]

        # generate some points first, say 10% of the sample
        if len(self.pattern) <= self.parameters.initial_sample_size:
            return self.first_order(voxel_centers)

        x, r = self.first_order(voxel_centers)

        # get nearest neighbor and calc its distance
        # to the current point
        index = self.pattern.nearest_neighbour(x, r)
        pairwise_distance = \
            np.linalg.norm(self.pattern.coordinates[index] - x)

        if pairwise_distance > self.parameters.cutoff_radius:
            return x, r

        # calculate the second order repulsion energy from distance
        e = self.energy_operator.second_order_potentials(pairwise_distance)

        best_p = x
        best_r = r
        best_e = e

        # metropolis hastings procedure for minimization of the
        # repulsion energy
        for _ in range(self.parameters.number_of_trials):

            # indepedent sampler inside the voxels of the current run
            trial_position, trial_radius = self.first_order(voxel_centers)

            index = self.pattern.nearest_neighbour(trial_position, trial_radius)
            pairwise_distance = \
                np.linalg.norm(self.pattern.coordinates[index] - trial_position)

            if pairwise_distance > self.parameters.cutoff_radius:
                return trial_position, trial_radius

            trial_energy = \
                self.energy_operator.second_order_potentials(pairwise_distance)

            logprob = self.parameters.beta * (e - trial_energy)

            if np.log(np.random.random()) < min(0, logprob):

                x = trial_position
                r = trial_radius
                e = trial_energy

            if e < best_e:

                best_e = e
                best_p = x
                best_r = r

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

        nonzero_intensity_groups = \
            [(i, v) for (i, v) in enumerate(intensity_per_group) if not np.isclose(v, 0.0)]

        for group_index, _ in nonzero_intensity_groups:

            group_total_counts = counts_per_group[group_index]
            # nuber_of_voxels = voxels_per_group[group_index]

            voxel_centers = vox_centers_per_group[group_index]

            k = 0
            while k < group_total_counts and len(self.pattern) < self._total_spheres:

                new_position, new_radius = self.method(voxel_centers)
                self.pattern.add(new_position, new_radius)

                k += 1

                # some logging for iteration info
                if len(self.pattern) % int(self._total_spheres * 0.1) == 0:
                    L.info('Current Number: {}'.format(len(self.pattern)))
