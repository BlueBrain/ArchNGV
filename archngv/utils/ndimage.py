"""Multi-dimensional image (e.g. volumes) utils."""
import logging
from typing import Optional, Tuple

import scipy.ndimage
from numpy.typing import NDArray

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
