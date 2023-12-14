"""
Generation algorithms for spatial point pattern
"""

import logging
from collections import namedtuple

import numpy as np

from archngv.building.cell_placement.pattern import SpatialSpherePattern

L = logging.getLogger(__name__)


PlacementParameters = namedtuple(
    "PlacementParameters",
    ["beta", "number_of_trials", "cutoff_radius", "initial_sample_size"],
)


class PlacementGenerator:
    """The workhorse of placement

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

    def __init__(
        self,
        parameters,
        total_spheres,
        voxel_data,
        energy_operator,
        index_list,
        soma_radius_distribution,
    ):
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
        """Check if
        1. position out of bounds
        2. sphere intersects with other_indexes list
        3. sphere intersects with other spheres in the pattern

        Args:
            trial_position: 1D array[float]
            trial_radius: float

        Returns: Bool
            True if collides or out of bounds
        """
        if not self.vdata.in_geometry(trial_position):
            return True

        if self.index_list:
            for static_index in self.index_list:
                if not static_index.sphere_empty(trial_position, trial_radius):
                    return True

        return self.pattern.is_intersecting(trial_position, trial_radius)

    def first_order(self, voxel_centers):
        """Sphere generation in the group of voxels with centers

        Args:
            voxel_centers: 2D array[float]
                Centers of intensity voxels

        Returns: 1D array[float], float
            New position and radius that is found by sampling the
            available space.
        """
        voxel_edge_length = self.vdata.voxelized_intensity.voxel_dimensions[0]

        while 1:
            new_position = proposal(voxel_centers, voxel_edge_length)
            new_radius = self.soma_proposal()

            if not self.is_colliding(new_position, new_radius):
                return new_position, new_radius

    def second_order(self, voxel_centers):
        """Sphere generation in the group with respect to interaction
        potentials. Valid is uniformly picked in the same
        intensity group using the first order approach and an extra
        metropolis hastings optimization step is performed in order
        to minimize the energy of the potential locally for each new
        sphere
        """
        # generate some points first without the second order interaction
        if len(self.pattern) <= self.parameters.initial_sample_size:
            return self.first_order(voxel_centers)

        current_position, current_radius = self.first_order(voxel_centers)

        pairwise_distance = self.pattern.distance_to_nearest_neighbor(
            current_position, self.parameters.cutoff_radius
        )

        if pairwise_distance > self.parameters.cutoff_radius:
            return current_position, current_radius

        current_energy = self.energy_operator.second_order_potentials(pairwise_distance)

        best_position = current_position
        best_radius = current_radius
        best_energy = current_energy

        # metropolis hastings procedure for minimization of the
        # repulsion energy
        for _ in range(self.parameters.number_of_trials):
            trial_position, trial_radius = self.first_order(voxel_centers)

            pairwise_distance = self.pattern.distance_to_nearest_neighbor(
                trial_position, self.parameters.cutoff_radius
            )

            if pairwise_distance > self.parameters.cutoff_radius:
                return trial_position, trial_radius

            trial_energy = self.energy_operator.second_order_potentials(pairwise_distance)

            logprob = self.parameters.beta * (current_energy - trial_energy)

            if np.log(np.random.random()) < min(0, logprob):
                current_position = trial_position
                current_radius = trial_radius
                current_energy = trial_energy

            if current_energy < best_energy:
                best_position = current_position
                best_radius = current_radius
                best_energy = current_energy

        return best_position, best_radius

    def run(self):
        """Create the population of spheres"""
        groups_generator = nonzero_intensity_groups(self.vdata.voxelized_intensity)

        for group_total_counts, voxel_centers in groups_generator:
            for _ in range(group_total_counts):
                if len(self.pattern) == self._total_spheres:
                    break

                new_position, new_radius = self.method(voxel_centers)
                self.pattern.add(new_position, new_radius)

                # some logging for iteration info
                if len(self.pattern) % 1000 == 0:
                    L.info("Current Number: %d", len(self.pattern))

        L.debug("Total spheres: %s", self._total_spheres)
        L.debug("Created spheres: %s", len(self.pattern))


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

    new_position = np.random.uniform(
        low=voxel_center - 0.5 * voxel_edge_length,
        high=voxel_center + 0.5 * voxel_edge_length,
        size=3,
    )

    return new_position


def voxel_grid_centers(voxel_grid):
    """
    Args:
        voxel_grid: VoxelData

    Returns: 2D array[float]
        Array of the centers of the grid voxels
    """
    unit_voxel_corners = np.indices(voxel_grid.shape, dtype=np.float32).reshape(3, -1).T
    return voxel_grid.indices_to_positions(unit_voxel_corners + 0.5)


def voxels_group_centers(labels, intensity):
    """Given the group labels of same intensity
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
    sorted_labels = labels.argsort(kind="stable")
    group_starts = np.searchsorted(labels[sorted_labels], np.unique(labels))

    voxel_centers = voxel_grid_centers(intensity)

    # return all the grouped voxel centers
    pairs = zip(group_starts[0:-1], group_starts[1::])

    groups = [voxel_centers[sorted_labels[i:j]] for i, j in pairs]
    groups += [voxel_centers[sorted_labels[group_starts[-1] : :]]]

    return groups


def counts_per_group(intensity_per_group, voxels_per_group, voxel_volume):
    """Returns the counts per group"""
    counts = 1e-9 * intensity_per_group * voxels_per_group * voxel_volume
    L.debug("Counts per group: %s, Total %s", counts, counts.sum())
    return counts.astype(np.int64)


def nonzero_intensity_groups(voxelized_intensity):
    """Generator that produces non zero intensity groups"""
    # group together voxels with identical values
    intensity_per_group, group_indices, voxels_per_group = np.unique(
        voxelized_intensity.raw, return_inverse=True, return_counts=True
    )

    vox_centers_per_group = voxels_group_centers(group_indices, voxelized_intensity)

    cnts_per_group = counts_per_group(
        intensity_per_group, voxels_per_group, voxelized_intensity.voxel_volume
    )

    for i, group_intensity in enumerate(intensity_per_group):
        if not np.isclose(group_intensity, 0.0):
            yield cnts_per_group[i], vox_centers_per_group[i]
