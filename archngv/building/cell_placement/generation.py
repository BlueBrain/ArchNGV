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

    def first_order(self, voxels, voxel_idx, probs):
        """Sphere generation based on voxels' probability of containing cells

        Args:
            voxels: 2D numpy array[int64] of shape (Nb_cell, 3) voxels (row,col, depth) indices
            voxel_idx: 1D numpy array[int64] of shape (Nb_cell,) voxels flat indices
            probs: 1D numpy array[int64] of shape (Nb_cell,) voxels probabilities of containing cells

        Returns: 1D array[float], float
            New position and radius that is found by sampling the
            available space.
        """
        # pick a first position
        for i in range(100000):
            # Get a random voxel id associated with the probabilities probs
            candidate_id = np.random.choice(voxel_idx, 1, p=probs)

            # Get a random position inside the candidate voxel
            new_position = self.vdata.voxelized_intensity.indices_to_positions(
                voxels[candidate_id] + np.random.random(3))[0]

            new_radius = self.soma_proposal()

            # Test that the new position is not colliding with other existing objets
            if not self.is_colliding(new_position, new_radius):
                return new_position, new_radius

        raise RuntimeError("Unable to generate a cell position that does not collide with other objets")


    def second_order(self, voxels, voxel_idx, probs):
        """Sphere generation in the group with respect to interaction
        potentials. Valid is uniformly picked in the same
        intensity group using the first order approach and an extra
        metropolis hastings optimization step is performed in order
        to minimize the energy of the potential locally for each new
        sphere
        """

        current_position, current_radius = self.first_order(voxels, voxel_idx, probs)

        # Return some points first without the second order interaction
        if len(self.pattern) <= self.parameters.initial_sample_size:
            return current_position, current_radius

        else:
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
                trial_position, trial_radius = self.first_order(voxels, voxel_idx, probs)

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

    def run(self, cell_count_per_voxel, cell_count):
        """Create the population of spheres"""
        density_factor = 1.
        if cell_count == 0:
            L.warning("Density resulted in zero cell counts.")
            return np.empty((0, 3), dtype=np.float32)
        # Get row/col/depth indices of none 0 density voxels
        voxel_ijk = np.nonzero(cell_count_per_voxel > 0)
        voxels = np.stack(voxel_ijk).transpose()
        # Get indices of none 0 density voxels
        voxel_idx = np.arange(len(voxel_ijk[0]))
        probs = 1.0 * cell_count_per_voxel[voxel_ijk] / np.sum(cell_count_per_voxel)

        while len(self.pattern) < self._total_spheres:
            new_position, new_radius = self.method(voxels, voxel_idx, probs)
            if new_position is None:
                print(f'No available pos for these voxels {voxel_centers}')
            else:
                self.pattern.add(new_position, new_radius)
            # some logging for iteration info
            if len(self.pattern) % 1000 == 0:
                L.info("Current Number: %d", len(self.pattern))

        L.debug("Total spheres: %s", self._total_spheres)
        L.debug("Created spheres: %s", len(self.pattern))


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
