"""Multi-dimensional image (e.g. volumes) utils."""
import logging
from typing import Generator, Optional, Tuple

import numpy
import scipy.ndimage
import voxcell
from numpy.typing import NDArray

from archngv.spatial.bounding_box import BoundingBox

L = logging.getLogger(__name__)


def connected_components(
    input_array: NDArray,
    structure: Optional[NDArray] = None,
) -> Tuple[NDArray, int]:
    """Compute the connected components of a multi-dimensional array.

    Args:
        input_array (numpy.array): Input array of n dimensions.
        structure (numpy.array):
            Optional connectivity matrix for considering neighbors in group. By default all the
            diagonals are considered.

            For example, for a 2x2 input array the default connectivity structure will be:

                [1 1 1]
                [1 1 1]
                [1 1 1]

    Returns:

        labeled_array:
            An integer ndarray where each unique feature in input has a unique label in the
            returned array.
        n_components: Number of components in the array.

    """
    if structure is None:
        structure = scipy.ndimage.generate_binary_structure(input_array.ndim, input_array.ndim)
    labeled_array, n_components = scipy.ndimage.label(input_array, structure=structure)

    L.info("Number of components labeled: %d", n_components)

    return labeled_array, n_components


def connected_components_sub_array(
    input_array: NDArray,
    structure: Optional[NDArray] = None,
    threshold: int = 10,
) -> Generator:
    """Connected components of a multidimensional array generator.
    Args:

        input_array (numpy.array): Input array of n dimensions.
        structure (numpy.array):
            Optional connectivity matrix for considering neighbors in group. By default all the
            diagonals are considered.

            For example, for a 2x2 input array the default connectivity structure will be:

                [1 1 1]
                [1 1 1]
                [1 1 1]
        threshold: The minimum number of pixels to be in a component to make it "legit"
            to yield it.
    Yields:
        NDArray:
            A sub array of the input_array
    """
    result, n_components = connected_components(input_array, structure)
    for i in range(n_components):
        mask = result == i + 1
        if numpy.count_nonzero(mask) >= threshold:
            yield input_array * mask


def map_positions_to_connected_components(
    positions: NDArray, voxel_data: voxcell.VoxelData, threshold: int = 10
):
    """
    Compute the connected components from the voxel data and return
     a list of non-connected component and their associated position ids.

    Args:
        positions: Array of xyz positions
        voxel_data: Voxel data of interest.
        threshold: The minimum number of pixels to be in a component to make it "legit"
             to yield it.

    Returns:
        a Tuple:
            - sub_bboxs
            - sub_dfs

    """
    ids = numpy.arange(len(positions), dtype=int)
    for mask in connected_components_sub_array(voxel_data.raw, threshold=threshold):
        bbox = BoundingBox.from_voxel_data_mask(
            mask,
            voxel_data.voxel_dimensions,
            voxel_data.offset,
        )

        mask = bbox.points_inside(positions)

        yield bbox, ids[mask]
