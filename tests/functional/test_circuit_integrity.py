"""This tests aims at checking the integrity of the circuit built from the snakemake."""

import numpy as np
import numpy.testing as npt

import voxcell

from archngv import NGVCircuit
import archngv.core.circuit as api
from archngv.core.datasets import Vasculature, MicrodomainTesselation, EndfootSurfaceMeshes


def test_circuit():
    circuit = NGVCircuit("build/ngv_config.json")

    # if a file is missing this will raise
    circuit.nodes
    circuit.edges

    # check accesses and simple values
    assert isinstance(circuit.nodes["All"], api.NGVNodes)
    assert isinstance(circuit.nodes["astrocytes"], api.Astrocytes)
    assert isinstance(circuit.nodes["vasculature"], api.Vasculature)
    assert isinstance(circuit.edges["All"], api.NGVEdges)
    assert isinstance(circuit.edges["glialglial"], api.GlialGlial)
    assert isinstance(circuit.edges["gliovascular"], api.GlioVascular)
    assert isinstance(circuit.edges["neuroglial"], api.NeuroGlial)
    assert isinstance(circuit.atlases, dict)

    assert circuit.neurons.size == 34
    assert circuit.astrocytes.size == 14
    assert circuit.vasculature.size == 10211
    assert circuit.neuronal_connectome.size == 641
    assert circuit.glialglial_connectome.size == 0  # no touches on functional
    assert circuit.neuroglial_connectome.size == 598
    assert circuit.gliovascular_connectome.size == 23
    assert len(circuit.atlases) == 2

    assert isinstance(circuit.astrocytes.microdomains, api.Microdomains)
    assert isinstance(circuit.astrocytes.microdomains.tesselation, MicrodomainTesselation)
    assert isinstance(circuit.astrocytes.microdomains.overlapping, MicrodomainTesselation)
    assert isinstance(circuit.gliovascular_connectome.surface_meshes, EndfootSurfaceMeshes)

    assert isinstance(circuit.atlases["intensity"], api.Atlas)
    assert isinstance(circuit.atlases["intensity"].get_atlas(), voxcell.VoxelData)
    assert isinstance(circuit.atlases["brain_regions"], api.Atlas)
    assert isinstance(circuit.atlases["brain_regions"].get_atlas(), voxcell.VoxelData)

    astrocytes = circuit.astrocytes
    # the morphologies.npy are created from the morphologies outputted from tns
    # (so translated / rotated). This verifies the positions / rotations are correctly propagated to
    # the sonata astrocyte node file.
    for astro_id in range(astrocytes.size):
        morphology_name = astrocytes.get(astro_id, "morphology")
        morphology_points = astrocytes.morphology.get(astro_id, transform=True).points
        expected_points = np.load("expected/morphologies/{}.npy".format(morphology_name))
        npt.assert_allclose(morphology_points, expected_points, atol=1e-4)


def test_vasculature_representations_consistency():

    circuit = NGVCircuit("build/ngv_config.json")

    from morphio.vasculature import Vasculature as mVasculature
    from archngv.core.datasets import Vasculature as sVasculature

    astrocytes = circuit.astrocytes
    gv_connectivity = circuit.gliovascular_connectome

    c_vasc = circuit.vasculature

    morphio_vasculature = mVasculature(c_vasc._extra_conf['vasculature_file'])
    sonata_vasculature = sVasculature.load_sonata('build/sonata/nodes/vasculature.h5')

    morphio_sections = morphio_vasculature.sections

    sonata_points = sonata_vasculature.points
    sonata_edges = sonata_vasculature.edges

    for aid in range(astrocytes.size):

        endfeet_ids = gv_connectivity.astrocyte_endfeet(aid)
        data = gv_connectivity.vasculature_sections_segments(endfeet_ids).to_numpy(dtype=np.int64)

        for edge_id, sec_id, seg_id in data:

            sonata_segment = sonata_points[sonata_edges[edge_id]]
            morphio_segment = morphio_sections[sec_id].points[seg_id: seg_id + 2]

            npt.assert_allclose(sonata_segment, morphio_segment)