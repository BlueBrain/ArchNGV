"""Circuit verification testing functions
"""
from pathlib import Path
from typing import Union

import numpy as np
from morphio import Morphology
from numpy import testing as npt

from archngv.core import circuit as api
from archngv.core import datasets
from archngv.exceptions import NGVError


def assert_circuit_integrity(circuit_or_config_path: Union[str, Path, api.NGVCircuit]):
    """Checks the integrity of the circuit with respect to the public api and and
    congruency of data.
    """
    circuit = (
        circuit_or_config_path
        if isinstance(circuit_or_config_path, api.NGVCircuit)
        else api.NGVCircuit(circuit_or_config_path)
    )

    _assert_instance(circuit, api.NGVCircuit)

    for check in [
        assert_astrocyte_population_integrity,
        assert_vasculature_population_integrity,
        assert_neuron_population_integrity,
        assert_astrocyte_data_integrity,
        assert_astrocyte_morphologies_integrity,
        assert_neuroglial_connectome_integrity,
        assert_gliovascular_connectome_integrity,
        assert_glialglial_connectome_integrity,
        assert_neuronal_connectome_integrity,
        assert_cross_population_integrity,
    ]:
        check(circuit)


def _assert_instance(obj, expected_class):
    """Checks that obj is of expected class"""
    assert isinstance(obj, expected_class), f"{obj} is not an instance of {expected_class}."


def assert_astrocyte_population_integrity(circuit: api.NGVCircuit):
    """Checks the integrity of the astrocyte node population"""

    _assert_instance(circuit.nodes["astrocytes"], api.Astrocytes)

    obj = circuit.astrocytes

    _assert_instance(obj, api.Astrocytes)
    _assert_instance(obj.microdomains, datasets.Microdomains)
    _assert_instance(obj.morph, api.AstrocytesMorphHelper)

    n_astrocytes = obj.size
    for astrocyte_id in (0, n_astrocytes // 2, n_astrocytes - 1):
        domain = obj.microdomains[astrocyte_id]

        assert len(domain.points) > 0
        assert len(domain.triangles) > 0

        filepath = obj.morph.get_filepath(astrocyte_id)

        assert filepath.endswith(".h5"), f"Astrocytes should be in hdf5 format. Found: {filepath}"

        id_from_filepath = int(str(Path(filepath).stem).replace("GLIA_", ""))

        assert id_from_filepath == astrocyte_id, (
            f"Astrocyte filepath does not correspond to its id:\n"
            f"Filepath ID: {id_from_filepath}\n"
            f"ID         : {astrocyte_id}"
        )


def assert_vasculature_population_integrity(circuit: api.NGVCircuit):
    """Checks the integrity of the vasculature node population"""

    _assert_instance(circuit.nodes["vasculature"], api.Vasculature)
    _assert_instance(circuit.vasculature, api.Vasculature)


def assert_neuron_population_integrity(circuit: api.NGVCircuit):
    """Checks the integrity of the neuronal node population"""

    _assert_instance(circuit.neurons, api.NGVNodes)


def assert_astrocyte_morphologies_integrity(circuit: api.NGVCircuit):
    """Checks that the files in the morphologies directory correspond to the ids in the astrocyte
    node population.
    """
    astrocytes = circuit.astrocytes

    morphologies_dir = astrocytes.config["alternate_morphologies"]["h5v1"]
    filenames = (
        path.stem
        for path in Path(morphologies_dir).glob("**/*.h5")
        if str(path.stem).startswith("GLIA_")
    )

    filepath_ids = set(int(name.replace("GLIA_", "")) for name in filenames)

    if not filepath_ids:
        raise AssertionError("There are no morphologies for the astrocytes population.")

    expected_ids = set(range(astrocytes.size))
    assert filepath_ids == expected_ids, (
        f"Morphologies do not correspond to the expected astrocyte ids.\n"
        f"May be that building hasn't finished successfully or it's or was partially executed.\n"
        f"Missing ids: {expected_ids - filepath_ids}"
    )


def assert_neuroglial_connectome_integrity(circuit: api.NGVCircuit):
    """Checks the integrity of the neuroglial connectivity."""

    _assert_instance(circuit.edges["neuroglial"], api.NeuroGlial)

    obj = circuit.neuroglial_connectome

    _assert_instance(obj, api.NeuroGlial)
    _assert_instance(obj.neurons, api.NGVNodes)
    _assert_instance(obj.astrocytes, api.NGVNodes)
    _assert_instance(obj.synapses, api.Neuronal)

    n_astrocytes = obj.astrocytes.size
    for astro_id in (0, n_astrocytes // 2, n_astrocytes - 1):
        obj.astrocyte_synapses(astro_id)
        obj.astrocyte_synapses_properties(astro_id)
        obj.connected_neurons(astro_id)

    n_neurons = obj.neurons.size
    for neuron_id in (0, n_neurons // 2, n_neurons - 1):
        obj.connected_astrocytes(neuron_id)


def assert_gliovascular_connectome_integrity(circuit: api.NGVCircuit):
    """Checks the integrity of the gliovascular connectivity."""

    _assert_instance(circuit.edges["gliovascular"], api.GlioVascular)

    obj = circuit.gliovascular_connectome

    _assert_instance(obj, api.GlioVascular)
    _assert_instance(obj.astrocytes, api.NGVNodes)
    _assert_instance(obj.vasculature, api.Vasculature)
    _assert_instance(obj.surface_meshes, datasets.EndfootSurfaceMeshes)

    n_astrocytes = obj.astrocytes.size
    for astrocyte_id in (0, n_astrocytes // 2, n_astrocytes - 1):
        obj.astrocyte_endfeet(astrocyte_id)
        obj.connected_vasculature(astrocyte_id)

    n_vasculature_segments = obj.vasculature.size

    for vasculature_segment_id in (0, n_vasculature_segments // 2, n_vasculature_segments - 1):
        obj.connected_astrocytes(vasculature_segment_id)
        obj.vasculature_endfeet(vasculature_segment_id)

    assert obj.size == len(obj.surface_meshes)

    n_endfeet = obj.size
    for endfoot_id in (0, n_endfeet // 2, n_endfeet - 1):
        obj.vasculature_surface_targets(endfoot_id)
        obj.vasculature_sections_segments(endfoot_id)


def assert_glialglial_connectome_integrity(circuit: api.NGVCircuit):
    """Checks the integrity of the glialglial connectivity."""

    _assert_instance(circuit.edges["glialglial"], api.GlialGlial)

    obj = circuit.glialglial_connectome

    _assert_instance(obj, api.GlialGlial)
    _assert_instance(obj.source, api.NGVNodes)
    _assert_instance(obj.target, api.NGVNodes)

    n_astrocytes = obj.target.size

    for astrocyte_id in (0, n_astrocytes // 2, n_astrocytes - 1):
        obj.astrocyte_gap_junctions(astrocyte_id)
        obj.astrocyte_astrocytes(astrocyte_id)


def assert_neuronal_connectome_integrity(circuit: api.NGVCircuit):
    """Checks integrity of the synaptic connectivity"""

    _assert_instance(circuit.neuronal_connectome, api.NGVEdges)


def assert_cross_population_integrity(circuit: api.NGVCircuit):
    """Checks that edge populations agree with each other."""

    ng_astrocytes = circuit.neuroglial_connectome.astrocytes
    gv_astrocytes = circuit.gliovascular_connectome.astrocytes

    assert ng_astrocytes.size == gv_astrocytes.size


def assert_astrocyte_data_integrity(circuit: api.NGVCircuit):
    """Checks that the astrocyte data from different sources agree with each other."""

    astrocytes = circuit.astrocytes
    ng_conn = circuit.neuroglial_connectome
    gv_conn = circuit.gliovascular_connectome
    endfeet_meshes = gv_conn.surface_meshes

    def get_valid_astrocyte():
        """Returns an astrocyte id which has endfeet, the surfaces of which have successfully
        grown.
        """
        for astrocyte_id in range(astrocytes.size):
            endfeet_ids = gv_conn.astrocyte_endfeet(astrocyte_id)

            if len(endfeet_ids) == 0:
                continue

            not_grown = [
                len(endfeet_meshes.mesh_points(endfoot_id)) == 0 for endfoot_id in endfeet_ids
            ]

            if any(not_grown):
                continue

            tripartite_ids = ng_conn.efferent_edges(astrocyte_id)

            if len(tripartite_ids) == 0:
                continue

            return astrocyte_id, endfeet_ids, tripartite_ids

        raise NGVError("No valid endfeet found.")

    astrocyte_id, endfeet_ids, tripartite_ids = get_valid_astrocyte()

    morph = Morphology(astrocytes.morph.get_filepath(astrocyte_id))
    astrocyte_position = astrocytes.positions(group=astrocyte_id).to_numpy()

    morphology_points = astrocyte_position + morph.points

    for endfoot_id in endfeet_ids:
        # In synthesis the surface_point is used as a target to grow the perivascular tree
        # towards that direction. The last point of that tree is always the target point.

        surface_point = gv_conn.get(
            endfoot_id, ["endfoot_surface_x", "endfoot_surface_y", "endfoot_surface_z"]
        ).to_numpy()

        assert np.isclose(morphology_points, surface_point, atol=1e-3).all(axis=1).any(), (
            f"A neurite must be connected to the surface point, but there isn't one.\n"
            f"Surface point: {surface_point}\n"
            f"Morphology points: {morphology_points}"
        )

        # In endfoot reconstruction the closest mesh vertex to the surface_point is found and
        # used as a starting vertex for the endfoot surface growth.

        mesh_points = endfeet_meshes.mesh_points(endfoot_id)
        distances = np.linalg.norm(mesh_points - surface_point, axis=1)

        assert np.any(distances < 5.0), (
            f"The surface point should be close to one of the mesh vertices, but it isn't.\n"
            f"Surface point      : {surface_point}\n"
            f"Endfoot mesh points: {mesh_points}"
        )

    # getting the section, segment, offset point from morphology should be equivalent to the
    # astrocyte_center_[x|y|z] properties

    df_annotations = ng_conn.get(
        tripartite_ids,
        properties=["astrocyte_section_id", "astrocyte_segment_id", "astrocyte_segment_offset"],
    )
    segment_points = ng_conn.get(
        tripartite_ids,
        properties=["astrocyte_center_x", "astrocyte_center_y", "astrocyte_center_z"],
    ).to_numpy()

    for i, (section_id, segment_id, offset) in enumerate(df_annotations.itertuples(index=False)):
        section = morph.section(section_id)
        seg_beg, seg_end = section.points[[segment_id, segment_id + 1]]

        axis = seg_end - seg_beg
        axis /= np.linalg.norm(axis)

        npt.assert_allclose(
            astrocyte_position + seg_beg + offset * axis,
            segment_points[i],
            atol=1e-5,
        )
