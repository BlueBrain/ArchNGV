"""This tests aims at checking the integrity of the circuit built from the snakemake."""

import numpy as np
import numpy.testing as npt

import voxcell

from archngv import NGVCircuit
from morphio import Morphology
import archngv.core.circuit as api
from archngv.core.datasets import Vasculature, MicrodomainTesselation, EndfootSurfaceMeshes

from bluepysnap.utils import IDS_DTYPE


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
    assert len(circuit.atlases) == 2

    assert isinstance(circuit.astrocytes.microdomains, api.Microdomains)
    assert isinstance(circuit.astrocytes.microdomains.tesselation, MicrodomainTesselation)
    assert isinstance(circuit.astrocytes.microdomains.overlapping, MicrodomainTesselation)

    assert isinstance(circuit.atlases["intensity"], api.Atlas)
    assert isinstance(circuit.atlases["intensity"].get_atlas(), voxcell.VoxelData)
    assert isinstance(circuit.atlases["brain_regions"], api.Atlas)
    assert isinstance(circuit.atlases["brain_regions"].get_atlas(), voxcell.VoxelData)


def test_neuroglial_connectome__property_dtypes():

    circuit = NGVCircuit("build/ngv_config.json")

    assert circuit.neuroglial_connectome.size == 598

    ng_conn = circuit.neuroglial_connectome

    npt.assert_equal(ng_conn.size, 598)

    prop_dtypes = {
        '@source_node': IDS_DTYPE,
        '@target_node': IDS_DTYPE,
        'synapse_id': np.uint64,
        'astrocyte_section_id': np.uint32,
        'astrocyte_segment_id': np.uint32,
        'astrocyte_segment_offset': np.float32,
        'astrocyte_section_pos': np.float32
    }

    expected_properties = set(prop_dtypes.keys())
    assert ng_conn.property_names == expected_properties, (ng_conn.property_names, expected_properties)

    for property_name, expected_dtype in prop_dtypes.items():

        arr = ng_conn.properties([0, 1], property_name)
        npt.assert_equal(arr.dtype, expected_dtype)


def test_neuroglial_connectome__annotation_equivalency():
    """Check that the section_id, segment_id, segment_offset annotation is equivalent to section_id, section_pos
    """
    circuit = NGVCircuit("build/ngv_config.json")
    ng_conn = circuit.neuroglial_connectome

    data = ng_conn.get(edge_ids=None, properties=['@source_node', 'astrocyte_section_id', 'astrocyte_segment_id', 'astrocyte_segment_offset', 'astrocyte_section_pos'])

    astro_ids = np.unique(data.loc[:, '@source_node'].to_numpy())
    astro_morphs = {int(i): Morphology(circuit.astrocytes.morph.get_filepath(int(i))) for i in astro_ids}

    for _, astrocyte_id, section_id, segment_id, segment_offset, expected_section_pos in data.itertuples():

        points = astro_morphs[astrocyte_id].sections[section_id].points
        segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)

        path_length = 0.0
        for i, length in enumerate(segment_lengths):
            if i < segment_id:
                path_length += length

        path_length += segment_offset

        # the section position is normalized by the section length
        section_position = path_length / segment_lengths.sum()

        npt.assert_allclose(section_position, expected_section_pos, atol=1e-6)


def test_gliovascular_connectome__property_dtypes():

    circuit = NGVCircuit("build/ngv_config.json")

    gv_conn = circuit.gliovascular_connectome

    npt.assert_equal(gv_conn.size, 23)
    assert isinstance(gv_conn.surface_meshes, EndfootSurfaceMeshes)

    prop_dtypes = {
        '@source_node' : IDS_DTYPE,
        '@target_node': IDS_DTYPE,
        'endfoot_id': np.uint64,
        'endfoot_surface_x': np.float32,
        'endfoot_surface_y': np.float32,
        'endfoot_surface_z': np.float32,
        'endfoot_compartment_length': np.float32,
        'endfoot_compartment_diameter': np.float32,
        'endfoot_compartment_perimeter': np.float32,
        'astrocyte_section_id': np.uint32,
        'vasculature_section_id': np.uint32,
        'vasculature_segment_id': np.uint32
    }

    expected_properties = set(prop_dtypes.keys())
    assert gv_conn.property_names == expected_properties, (gv_conn.property_names, expected_properties)

    circuit_dtypes = gv_conn.property_dtypes

    for property_name, expected_dtype in prop_dtypes.items():
        npt.assert_equal(circuit_dtypes[property_name], expected_dtype, err_msg=f'Property: {property_name}')



def test_vasculature_representations_consistency():
    """Test that it is equivalent to get the segment coordinates
    via the sonata and morphio representations
    """
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
