""" Astrocyte post synthesis properties
"""
import os

import morphio
import numpy as np

from archngv.core.datasets import CellData


def _extract_section_perimeters(morphology):
    """
    def equivalent_perimeters(section):
        return 1.0

        points, perimeters = section.points, section.perimeters

        radii = perimeters / (2. * np.pi)
        lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)

        areas = 0.5 * (perimeters[1:] + perimeters[:-1])
                    * np.sqrt(lengths ** 2 + (radii[1:]
                    - radii[:-1]) ** 2)

        return areas.sum() / lengths.sum()
    """

    def equivalent_perimeters(_):
        return 1.0

    return np.fromiter(map(equivalent_perimeters, morphology.iter()), dtype=np.float32)


def create_astrocyte_properties(astrocyte_index, paths):
    """Generates annotations for endfeet and synapses, endfoot compartment information
    and astrocyte section perimeters.
    """
    cell_name = CellData(paths.cell_data).get_properties("morphology", astrocyte_index)[0][0]

    # For the annotations the morphology should be in readonly mode. If it is mutated
    # for any reason there will be reordering of the sections ids and thus the
    # annotations would be invalid
    morph_filepath = os.path.join(paths.morphology_directory, cell_name + ".h5")
    morphology = morphio.Morphology(morph_filepath, options=morphio.Option.nrn_order)

    # pylint: disable=too-many-arguments
    properties = {"name": cell_name}
    properties["section_perimeters"] = _extract_section_perimeters(morphology)

    return properties
