"""Endfeet properties to be added to the gliovascular edge population file"""
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Tuple

import numpy as np
import pandas as pd

from archngv.exceptions import NGVError

if TYPE_CHECKING:
    from vascpy import PointVasculature

    from archngv.core.datastructure import CellData, EndfeetMeshes, GliovascularConnectivity


def endfeet_mesh_properties(
    seed: int,
    astrocytes: "CellData",
    gliovascular_connectivity: "GliovascularConnectivity",
    vasculature: "PointVasculature",
    endfeet_meshes: "EndfeetMeshes",
    morph_dir: Path,
    map_function: Callable[[Any, Any], Any],
) -> pd.DataFrame:
    """Generates the endfeet properties that required for the astrocyte morphologies and
    endfeet meshes. These properties will then be added to the GliovascularConnectivity edges
    as properties along with the previous ones, calculated at the first stages of the framework

    Args:
        seed: The seed for the random generator
        astrocytes: astrocytes population
        gv_connectivity: edges population
        vasculature: sonata vasculature
        endfeet_meshes: Meshes for endfeet
        morph_dir: Morphology directory
        map_func: map function to run the processing

    Returns:
        dict: A dictionary with additional endfeet properties:
            ids (np.ndarray): (N,) int array of endfeet ids
            astrocyte_section_id (np.ndarray): int array of astrocyte morphology section id that
                connects to the surface of the vasculature
            endfoot_compartment_length (np.ndarray): (N,) float array of compartment lengths
            endfoot_compartment_diameter (np.ndarray): (N,) float array of compartment diameters
            endfoot_compartmen_perimeter (np.ndarray): (N,) float array of compartment perimeters
    """
    endfeet_ids = gliovascular_connectivity.get_property("endfoot_id")
    n_endfeet = len(endfeet_ids)

    if not np.array_equal(endfeet_ids, np.arange(n_endfeet)):
        raise NGVError("endfeet_ids should be a contiguous array from 0 to number of endfeet")

    properties = {
        "astrocyte_section_id": np.empty(n_endfeet, dtype=np.uint32),
        "endfoot_compartment_length": np.empty(n_endfeet, dtype=np.float32),
        "endfoot_compartment_diameter": np.empty(n_endfeet, dtype=np.float32),
        "endfoot_compartment_perimeter": np.empty(n_endfeet, dtype=np.float32),
    }

    it_results = map_function(
        GliovascularWorker(seed),
        _dispatch_endfeet_data(
            astrocytes,
            gliovascular_connectivity,
            vasculature,
            endfeet_meshes,
            morph_dir,
        ),
    )

    for ids, section_ids, lengths, diameters, perimeters in it_results:
        properties["astrocyte_section_id"][ids] = section_ids
        properties["endfoot_compartment_length"][ids] = lengths
        properties["endfoot_compartment_diameter"][ids] = diameters
        properties["endfoot_compartment_perimeter"][ids] = perimeters

    return properties


class GliovascularWorker:
    """Endfeet properties parallel helper"""

    def __init__(self, seed):
        self._seed = seed

    def __call__(self, data):
        """
        Args:
            data (dict)
        """
        np.random.seed(hash((self._seed, data["index"])) % (2**32))
        return _endfeet_properties_from_astrocyte(data)


def _dispatch_endfeet_data(
    astrocytes: "CellData",
    gv_connectivity: "GliovascularConnectivity",
    vasculature: "PointVasculature",
    endfeet_meshes: "EndfeetMeshes",
    morph_dir: Path,
) -> Iterator[dict]:
    """Dispatches data for parallel worker
    Args:
        astrocytes: Astrocyte population
        gv_connectivity: Edges population
        vasculature: Sonata vasculature
        endfeet_meshes: The data for the endfeet meshes
        morph_dir: Path to morphology directory

    Yields:
        data: The following pairs:
            index (float): astrocyte index
            endfeet_ids (np.ndarray): (N,) Endfeet ids for astrocyte index
            endfeet_surface_targets (np.ndarray): (N, 3) Surface starting points
                of endfeet
            endfeet_meshes (List(namedtuple)): (N,) Meshes of endfeet surfaces with:
                index (int): endfoot index
                points (np.ndarray): mesh points
                triangles (np.ndarray): mesh triangles
                area (float): mesh surface area
                thickness (float): mesh thickness
            morphology_path (str): Path to astrocyte morphology
            morphology_position (np.ndarray): (3,) Position of astrocyte
            vasculature_segments (np.ndarray): (N, 2, 3) Vasculature segments per
                endfoot
    """
    vasculature_points = vasculature.points
    vasculature_edges = vasculature.edges

    for astro_id in range(len(astrocytes)):
        endfeet_ids = gv_connectivity.astrocyte_endfeet(astro_id)

        # no endfeet, no processing to do
        if endfeet_ids.size == 0:
            continue

        morphology_name = astrocytes.get_property("morphology", ids=astro_id)[0]
        morphology_path = str(Path(morph_dir, morphology_name + ".h5"))
        morphology_pos = astrocytes.positions(index=astro_id)[0]

        vasc_segment_ids = gv_connectivity.vasculature_sections_segments(endfeet_ids)[:, 0]
        vasc_segments = vasculature_points[vasculature_edges[vasc_segment_ids]]

        endfeet_surface_targets = gv_connectivity.vasculature_surface_targets(endfeet_ids)

        yield {
            "index": astro_id,
            "endfeet_ids": endfeet_ids,
            "endfeet_surface_targets": endfeet_surface_targets,
            "endfeet_meshes": endfeet_meshes[endfeet_ids],
            "morphology_path": morphology_path,
            "morphology_position": morphology_pos,
            "vasculature_segments": vasc_segments,
        }


def _endfeet_properties_from_astrocyte(data: dict) -> Tuple[np.ndarray, ...]:
    """Calculates data for one astrocyte
    Args:
        data: Input data dict. See _dispatch_data for the dict key, values
    Returns:
        tuple:
            endfeet_ids: (N,) int array of endfeet ids
            astrocyte_section_ids: (N,) int array of astrocyte section ids
            lengths: (N,) float array of endfeet compartment lengths
            diameters: (N,) float array of endfeet compartment diameters
            perimeters: (N,) float array of endfeet compartment perimeters
    """
    from archngv.app.utils import readonly_morphology
    from archngv.building.morphology_synthesis.annotation import annotate_endfoot_location
    from archngv.building.morphology_synthesis.endfoot_compartment import (
        create_endfeet_compartment_data,
    )

    morphology = readonly_morphology(data["morphology_path"], data["morphology_position"])
    try:
        astrocyte_section_ids = annotate_endfoot_location(morphology, data["endfeet_surface_targets"])
    except ValueError:
        print(f'DEBUG data["morphology_path"] {data["morphology_path"]}')
        raise ValueError(f'DEBUG data["morphology_path"] {data["morphology_path"]}')

    lengths, diameters, perimeters = create_endfeet_compartment_data(
        data["vasculature_segments"],
        data["endfeet_surface_targets"],
        data["endfeet_meshes"],
    )

    return data["endfeet_ids"], astrocyte_section_ids, lengths, diameters, perimeters
