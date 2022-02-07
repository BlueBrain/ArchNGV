"""ngv cli"""
# pylint: disable=too-many-statements
from pathlib import Path
import click

import numpy
import voxcell

from archngv.app.utils import load_yaml, write_json
from archngv.exceptions import NGVError


@click.command()
@click.option('--bioname', help='Path to bioname folder', required=True)
@click.option('-o', '--output', help='Path to output file (JSON)', required=True)
def ngv_config(bioname, output):
    """Build NGV SONATA config"""
    from archngv.building.config import build_ngv_config

    bioname = Path(bioname).resolve()
    manifest = load_yaml(Path(bioname, "MANIFEST.yaml"))

    # resolving relative paths is needed for the cases where the executor is not
    # at the same place as the data. For example, pytest creates an isolated filesystem
    # and changes the current directory to that. Relative paths without resolution would
    # me resolved with respect to that isolated filesystem, producing invalid paths
    for key, value in manifest["common"].items():
        if isinstance(value, str) and not Path(value).is_absolute():
            manifest["common"][key] = str(Path(bioname, value).resolve())

    # TODO: parameters should not be part of the sonata config, remove in the future
    for rule in ["cell_placement", "microdomains", "gliovascular_connectivity", "synthesis"]:
        manifest[rule] = load_yaml(Path(bioname, f"{rule}.yaml"))

    manifest["synthesis"]["endfeet_area_reconstruction"] = load_yaml(Path(bioname, "endfeet_area.yaml"))

    write_json(
        filepath=output,
        data=build_ngv_config(root_dir=bioname, manifest=manifest)
    )


# pylint: disable=redefined-builtin
@click.command()
@click.option("-i", "--input", help="Path to input SONATA Nodes HDF5", required=True)
@click.option("--hoc", help="HOC template file name", required=True)
@click.option("-o", "--output", help="Path to output SONATA Nodes HDF5", required=True)
def assign_emodels(input, hoc, output):
    """Assign `model_template` attribute to node population"""
    emodels = voxcell.CellCollection.load_sonata(input)
    cols = list(emodels.properties)
    emodels.properties['model_template'] = f'hoc:{hoc}'
    emodels.properties = emodels.properties.drop(columns=cols)
    emodels.save_sonata(output)


@click.command()
@click.option("--config", help="Path to astrocyte placement YAML config", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--vasculature", help="Path to vasculature node population", default=None, required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output SONATA nodes file", required=True)
def cell_placement(config, atlas, atlas_cache, vasculature, seed, output):
    """
    Generate astrocyte positions and radii inside the bounding box of the vasculature dataset.

    Astrocytes are placed without colliding with the vasculature geometry or with other astrocytic
    somata.
    """
    # pylint: disable=too-many-locals

    from voxcell.nexus.voxelbrain import Atlas
    from vasculatureapi import PointVasculature
    from spatial_index import SphereIndex

    from archngv.building.cell_placement.positions import create_positions
    from archngv.building.exporters.node_populations import export_astrocyte_population
    from archngv.building.checks import assert_bbox_alignment

    from archngv.spatial import BoundingBox
    from archngv.app.logger import LOGGER

    numpy.random.seed(seed)
    LOGGER.info("Seed: %d", seed)

    config = load_yaml(config)

    atlas = Atlas.open(atlas, cache_dir=atlas_cache)
    voxelized_intensity = atlas.load_data(config['density'])
    voxelized_bnregions = atlas.load_data('brain_regions')

    assert numpy.issubdtype(voxelized_intensity.raw.dtype, numpy.floating)

    spatial_indexes = []
    if vasculature is not None:

        vasc = PointVasculature.load_sonata(vasculature)

        assert_bbox_alignment(
            BoundingBox.from_points(vasc.points),
            BoundingBox(voxelized_intensity.bbox[0],
                        voxelized_intensity.bbox[1])
        )

        spatial_indexes.append(SphereIndex(vasc.points, 0.5 * vasc.diameters))

    LOGGER.info('Generating cell positions / radii...')
    somata_positions, somata_radii = create_positions(
        config,
        voxelized_intensity,
        voxelized_bnregions,
        spatial_indexes=spatial_indexes
    )

    cell_names = ['GLIA_{:013d}'.format(index) for index in range(len(somata_positions))]

    LOGGER.info('Export to CellData...')
    export_astrocyte_population(output, cell_names, somata_positions, somata_radii, mtype="ASTROCYTE")

    LOGGER.info('Done!')


@click.command()
@click.option("--somata-file", help="Path to sonata somata file", required=True)
@click.option("--emodels-file", help="Path to sonata emodels file", required=True)
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def finalize_astrocytes(somata_file, emodels_file, output):
    """Build the finalized astrocyte node population by merging the different astrocyte properties
    """
    from archngv.core.constants import Population
    somata = voxcell.CellCollection.load(somata_file)
    emodels = voxcell.CellCollection.load(emodels_file)
    somata.properties['model_template'] = emodels.properties['model_template']
    somata.population_name = Population.ASTROCYTES
    somata.save_sonata(output)


@click.command()
@click.option("--config", help="Path to astrocyte microdomains YAML config", required=True)
@click.option("--astrocytes", help="Path to the sonata file with astrocyte's positions and radii", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output-dir", help="Path to output MVD3", required=True)
def build_microdomains(config, astrocytes, atlas, atlas_cache, seed, output_dir):
    """Generate astrocyte microdomain tesselation as a partition of space into convex polygons.
    """
    # pylint: disable=missing-docstring,too-many-locals
    from scipy import stats
    from voxcell.nexus.voxelbrain import Atlas

    from archngv.building.exporters.export_microdomains import export_structure
    from archngv.building.microdomain.generation import generate_microdomain_tesselation
    from archngv.building.microdomain.generation import convert_to_overlappping_tesselation

    from archngv.spatial import BoundingBox

    from archngv.app.logger import LOGGER
    from archngv.app.utils import ensure_dir

    def _output_path(filename):
        return str(Path(output_dir, filename))

    LOGGER.info('Seed: %d', seed)
    numpy.random.seed(seed)

    config = load_yaml(config)

    atlas = Atlas.open(atlas, cache_dir=atlas_cache)
    bbox = atlas.load_data('brain_regions').bbox
    bounding_box = BoundingBox(bbox[0], bbox[1])

    astrocytes = voxcell.CellCollection.load_sonata(astrocytes)
    astrocyte_positions = astrocytes.positions
    astrocyte_radii = astrocytes.properties['radius'].to_numpy()

    ensure_dir(output_dir)

    LOGGER.info('Generating microdomains...')
    microdomains = generate_microdomain_tesselation(
        astrocyte_positions, astrocyte_radii, bounding_box
    )

    LOGGER.info('Export microdomains...')
    export_structure(_output_path('microdomains.h5'), microdomains)

    LOGGER.info('Generating overlapping microdomains...')
    overlap_distr = config['overlap_distribution']['values']
    overlap_distribution = stats.norm(loc=overlap_distr[0], scale=overlap_distr[1])
    overlapping_microdomains = convert_to_overlappping_tesselation(microdomains, overlap_distribution)

    LOGGER.info('Export overlapping microdomains...')
    export_structure(_output_path('overlapping_microdomains.h5'), overlapping_microdomains)

    LOGGER.info('Done!')


@click.group()
def gliovascular_group():
    """Gliovascular group"""


@gliovascular_group.command(name="connectivity")
@click.option("--config", help="Path to astrocyte placement YAML config", required=True)
@click.option("--astrocytes", help="Path to the sonata file with astrocyte's positions", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--vasculature", help="Path to vasculature sonata dataset", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--output", help="Path to output edges HDF5 (data)", required=True)
def build_gliovascular_connectivity(config, astrocytes, microdomains, vasculature, seed, output):
    """
    Build connectivity between astrocytes and the vasculature graph.
    """
    # pylint: disable=redefined-argument-from-local,too-many-locals
    from vasculatureapi import PointVasculature

    from archngv.core.datasets import MicrodomainTesselation

    from archngv.building.connectivity.gliovascular_generation import generate_gliovascular
    from archngv.building.exporters.edge_populations import gliovascular_connectivity

    from archngv.app.logger import LOGGER

    LOGGER.info('Seed: %d', seed)
    numpy.random.seed(seed)

    LOGGER.info('Generating gliovascular connectivity...')

    astrocytes = voxcell.CellCollection.load_sonata(astrocytes)

    (
        endfoot_surface_positions,
        endfeet_to_astrocyte_mapping,
        endfeet_to_vasculature_mapping
    ) = generate_gliovascular(
        cell_ids=numpy.arange(len(astrocytes), dtype=numpy.int64),
        astrocytic_positions=astrocytes.positions,
        astrocytic_domains=MicrodomainTesselation(microdomains),
        vasculature=PointVasculature.load_sonata(vasculature),
        params=load_yaml(config)
    )

    LOGGER.info('Exporting sonata edges...')
    gliovascular_connectivity(
        output,
        astrocytes,
        voxcell.CellCollection.load_sonata(vasculature),
        endfeet_to_astrocyte_mapping,
        endfeet_to_vasculature_mapping,
        endfoot_surface_positions,
    )

    LOGGER.info("Done!")


def _endfeet_properties_from_astrocyte(data):
    """Calculates data for one astrocyte
    Args:
        data (dict): Input data dict. See _dispatch_data for the dict key, values
    Returns:
        tuple:
            endfeet_ids (np.ndarray): (N,) int array of endfeet ids
            astrocyte_section_ids (np.ndarray): (N,) int array of astrocyte section ids
            lengths (np.ndarray): (N,) float array of endfeet compartment lengths
            diameters (np.ndarray): (N,) float array of endfeet compartment diameters
            perimeters (np.ndarray): (N,) float array of endfeet compartment perimeters
    """
    from archngv.building.morphology_synthesis.annotation import annotate_endfoot_location
    from archngv.building.morphology_synthesis.endfoot_compartment import create_endfeet_compartment_data
    from archngv.app.utils import readonly_morphology

    morphology = readonly_morphology(data['morphology_path'], data['morphology_position'])
    astrocyte_section_ids = annotate_endfoot_location(morphology, data['endfeet_surface_targets'])

    lengths, diameters, perimeters = create_endfeet_compartment_data(
        data['vasculature_segments'],
        data['endfeet_surface_targets'],
        data['endfeet_meshes']
    )

    return data['endfeet_ids'], astrocyte_section_ids, lengths, diameters, perimeters


class GliovascularWorker:
    """Endfeet properties parallel helper"""
    def __init__(self, seed):
        self._seed = seed

    def __call__(self, data):
        """
        Args:
            data (dict)
        """
        numpy.random.seed(hash((self._seed, data['index'])) % (2 ** 32))
        return _endfeet_properties_from_astrocyte(data)


def _apply_parallel_func(func, data_generator):
    """Apply the function on the data generator in parallel and yield the results"""
    import multiprocessing
    from archngv.app.logger import LOGGER
    n_cores = multiprocessing.cpu_count()
    LOGGER.info('Run in parallel enabled. N cores: %d', n_cores)
    with multiprocessing.Pool(n_cores) as p:
        for result in p.imap(func, data_generator):
            yield result


def _dispatch_endfeet_data(astrocytes, gv_connectivity, vasculature, endfeet_meshes, morph_dir):
    """Dispatches data for parallel worker
    Args:
        astrocytes (CellData): Astrocyte population
        gv_connectivity (GliovascularConnectivity): Edges population
        vasculature (Vasculature): Sonata vasculature
        endfeet_meshes (EndfeetSurfaceMeshes): The data for the endfeet meshes
        morph_dir (str): Path to morphology directory

    Yields:
        data (dict): The following pairs:
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

        morphology_name = astrocytes.get_property('morphology', ids=astro_id)[0]
        morphology_path = str(Path(morph_dir, morphology_name + '.h5'))
        morphology_pos = astrocytes.positions(index=astro_id)[0]

        vasc_segment_ids = gv_connectivity.vasculature_sections_segments(endfeet_ids)[:, 0]
        vasc_segments = vasculature_points[vasculature_edges[vasc_segment_ids]]

        endfeet_surface_targets = gv_connectivity.vasculature_surface_targets(endfeet_ids)

        yield {
            'index': astro_id,
            'endfeet_ids': endfeet_ids,
            'endfeet_surface_targets': endfeet_surface_targets,
            'endfeet_meshes': endfeet_meshes[endfeet_ids],
            'morphology_path': morphology_path,
            'morphology_position': morphology_pos,
            'vasculature_segments': vasc_segments
        }


def _endfeet_properties(seed, astrocytes, gv_connectivity,
                        vasculature, endfeet_meshes, morph_dir, map_func):
    """Generates the endfeet properties that required the astrocyte morphologies and
    endfeet areas. These properties will then be added to the GliovascularConnectivity edges
    as properties along with the previous ones, calculated at the first stages of the framework

    Args:
        seed (int): The seed for the random generator
        astrocytes (CellData): astrocytes population
        gv_connectivity: (GliovascularConnectivity): edges population
        vasculature (Vasculature): sonata vasculature
        endfeet_meshes (EndfootSurfaceMeshes): Meshes for endfeet
        morph_dir (str): Morphology directory
        map_func (Callable): map function to run the processing, parallel or not

    Returns:
        dict: A dictionary with additional endfeet properties:
            ids (np.ndarray): (N,) int array of endfeet ids
            astrocyte_section_id (np.ndarray): int array of astrocyte morphology section id that
                connects to the surface of the vasculature
            endfoot_compartment_length (np.ndarray): (N,) float array of compartment lengths
            endfoot_compartment_diameter (np.ndarray): (N,) float array of compartment diameters
            endfoot_compartmen_perimeter (np.ndarray): (N,) float array of compartment perimeters
    """
    endfeet_ids = gv_connectivity.get_property('endfoot_id')
    n_endfeet = len(endfeet_ids)

    if not numpy.array_equal(endfeet_ids, numpy.arange(n_endfeet)):
        raise NGVError('endfeet_ids should be a contiguous array from 0 to number of endfeet')

    properties = {
        'astrocyte_section_id': numpy.empty(n_endfeet, dtype=numpy.uint32),
        'endfoot_compartment_length': numpy.empty(n_endfeet, dtype=numpy.float32),
        'endfoot_compartment_diameter': numpy.empty(n_endfeet, dtype=numpy.float32),
        'endfoot_compartment_perimeter': numpy.empty(n_endfeet, dtype=numpy.float32)
    }

    it_results = map_func(
        GliovascularWorker(seed),
        _dispatch_endfeet_data(astrocytes, gv_connectivity, vasculature, endfeet_meshes, morph_dir)
    )

    for ids, section_ids, lengths, diameters, perimeters in it_results:

        properties['astrocyte_section_id'][ids] = section_ids
        properties['endfoot_compartment_length'][ids] = lengths
        properties['endfoot_compartment_diameter'][ids] = diameters
        properties['endfoot_compartment_perimeter'][ids] = perimeters

    return properties


@gliovascular_group.command(name="finalize")
@click.option("--input-file", help="Path to sonata edges file (HDF5)", required=True)
@click.option("--output-file", help="Path to sonata edges file (HDF5)", required=True)
@click.option("--astrocytes", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--endfeet-areas", help="Path to HDF5 endfeet areas", required=True)
@click.option("--vasculature-sonata", help="Path to nodes for vasculature (HDF5)", required=True)
@click.option("--morph-dir", help="Path to morphology folder", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--parallel", help="Parallelize with 'multiprocessing'", is_flag=True, default=False)
def gliovascular_finalize(
        input_file, output_file, astrocytes, endfeet_areas, vasculature_sonata, morph_dir, seed, parallel):
    """
    Finalizes gliovascular connectivity. It needs to be ran after synthesis and endfeet area growing.
    It copies to the input GliovascularConnectivity population and adds the following edge properties:

        - astrocyte_section_id
            The last section id of the astrocytic morphology that connects to the endfoot locaiton on
            the vascular surface

        - endfoot_compartment_length
            The extent of the endfoot across the medial axis of the segment that is located

        - endfoot_compartment_diameter
            A diameter value so that diameter * length = volume of endfoot

        - endfoot_compartment_perimeter
            A perimeter value so that perimeter * length = area of endfoot
    """
    import shutil
    from vasculatureapi import PointVasculature
    from archngv.core.datasets import CellData, GliovascularConnectivity, EndfootSurfaceMeshes
    from archngv.building.exporters.edge_populations import add_properties_to_edge_population
    from archngv.app.utils import apply_parallel_function

    gv_connectivity = GliovascularConnectivity(input_file)

    properties = _endfeet_properties(
        seed=seed,
        astrocytes=CellData(astrocytes),
        gv_connectivity=gv_connectivity,
        vasculature=PointVasculature.load_sonata(vasculature_sonata),
        endfeet_meshes=EndfootSurfaceMeshes(endfeet_areas),
        morph_dir=morph_dir,
        map_func=apply_parallel_function if parallel else map
    )
    # copy gv file to the output location
    shutil.copyfile(input_file, output_file)

    # add the new properties to the copied out file
    add_properties_to_edge_population(output_file, gv_connectivity.name, properties)


@click.group()
def neuroglial_group():
    """Generate neuroglial (N-G) connectivity"""


@neuroglial_group.command(name="connectivity")
@click.option("--neurons", help="Path to neuron node population (SONATA Nodes)", required=True)
@click.option("--astrocytes", help="Path to astrocyte node population (SONATA Nodes)", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--neuronal-connectivity", help="Path to neuron-neuron sonata edge file", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output file (SONATA Edges HDF5)", required=True)
def build_neuroglial_connectivity(
        neurons, astrocytes, microdomains, neuronal_connectivity, seed, output):
    """ Generate connectivity between neurons (N) and astrocytes (G) """
    # pylint: disable=redefined-argument-from-local,too-many-locals

    from archngv.core.datasets import (
        NeuronalConnectivity,
        MicrodomainTesselation
    )
    from archngv.building.connectivity.neuroglial_generation import generate_neuroglial
    from archngv.building.exporters.edge_populations import neuroglial_connectivity

    from archngv.app.logger import LOGGER
    numpy.random.seed(seed)

    astrocytes_data = voxcell.CellCollection.load(astrocytes)

    LOGGER.info('Generating neuroglial connectivity...')

    microdomains = MicrodomainTesselation(microdomains)

    data_iterator = generate_neuroglial(
        astrocytes=astrocytes_data,
        microdomains=microdomains,
        neuronal_connectivity=NeuronalConnectivity(neuronal_connectivity)
    )

    LOGGER.info('Exporting the per astrocyte files...')
    neuroglial_connectivity(
        data_iterator,
        neurons=voxcell.CellCollection.load(neurons),
        astrocytes=astrocytes_data,
        output_path=output
    )

    LOGGER.info("Done!")


def _properties_from_astrocyte(data):
    """Processes one astrocyte and returns annotation properties
    Args:
        data (dict): Dictionary with the following keys
            - index: astrocyte index
            - neurogial_connectivity: Path to ng connectivity sonata file
            - synaptic_data: Path to synaptic data sonata file
            - morphology_path: Path to morphology h5 file
            - morphology_position: Coordinates of morphology soma position

    Returns:
        tuple:
            connection_ids (np.ndarray): sonata edge ids
            section_ids (np.ndarray): Morphology section ids
            segment_ids: Morphology segment ids
            segment_offsets: Morphology segment offsets
    """
    from archngv.building.morphology_synthesis.annotation import annotate_synapse_location
    from archngv.core.datasets import NeuroglialConnectivity, NeuronalConnectivity
    from archngv.app.utils import readonly_morphology

    astrocyte_index = data['index']

    ng_connectivity = NeuroglialConnectivity(data['neuroglial_connectivity'])
    connection_ids = ng_connectivity.astrocyte_neuron_connections(astrocyte_index)

    if connection_ids.size == 0:
        return None

    synapse_ids = ng_connectivity.neuronal_synapses(connection_ids)

    synaptic_data = NeuronalConnectivity(data['synaptic_data'])
    synapse_positions = synaptic_data.synapse_positions(synapse_ids)

    morphology = readonly_morphology(data['morphology_path'], data['morphology_position'])
    locations_dataframe = annotate_synapse_location(morphology, synapse_positions)

    return connection_ids, locations_dataframe


class NeuroglialWorker:
    """Neuroglial properties helper"""
    def __init__(self, seed):
        self._seed = seed

    def __call__(self, data):

        seed = hash((self._seed, data['index'])) % (2 ** 32)
        numpy.random.seed(seed)

        return _properties_from_astrocyte(data)


def _dispatch_neuroglial_data(astrocytes, paths):

    for astro_id in range(len(astrocytes)):

        morphology_name = astrocytes.get_property('morphology', ids=astro_id)[0]
        morphology_path = str(Path(paths['morph_dir'], morphology_name + '.h5'))
        morphology_pos = astrocytes.positions(index=astro_id)[0]

        data = {
            'index': astro_id,
            'morphology_path': morphology_path,
            'morphology_position': morphology_pos
        }

        data.update(paths)

        yield data


def _neuroglial_properties(seed, astrocytes, n_connections, paths, map_func):
    """
    Args:
        seed (int): Random generator's seed
        astrocytes (CellData): node population of astrocytes
        n_connections (int): number of astrocyte-neuron connections
        paths (dict): dictionary with paths
        map_func (Callable): parallelization function

    Returns:
        properties (dict): Dictionary with string keys
            astrocyte_section_id (np.ndarray): Array of ints corresponding to the
                astrocyte section id associated with each connected synapse
            astrocyte_segment_id (np.ndarray): Array of ints corresponding to the
                astrocyte segment id associated with each connected synapse
            astrocyte_segment_offset (np.ndarray): Array of floats corresponding
                to the segment offset associated with each connected synapse
    """

    properties = {
        'astrocyte_section_id': numpy.empty(n_connections, dtype=numpy.uint32),
        'astrocyte_segment_id': numpy.empty(n_connections, dtype=numpy.uint32),
        'astrocyte_segment_offset': numpy.empty(n_connections, dtype=numpy.float32),
        'astrocyte_section_pos': numpy.empty(n_connections, dtype=numpy.float32)
    }

    it_results = filter(
        lambda result: result is not None,
        map_func(NeuroglialWorker(seed), _dispatch_neuroglial_data(astrocytes, paths))
    )

    for ids, df_locations in it_results:

        properties['astrocyte_section_id'][ids] = df_locations.section_id
        properties['astrocyte_segment_id'][ids] = df_locations.segment_id
        properties['astrocyte_segment_offset'][ids] = df_locations.segment_offset
        properties['astrocyte_section_pos'][ids] = df_locations.section_position

    return properties


@neuroglial_group.command(name="finalize")
@click.option("--input-file", help="Path to input file (SONATA Edges HDF5)", required=True)
@click.option("--output-file", help="Path to output file (SONATA Edges HDF5)", required=True)
@click.option("--astrocytes", help="Path to astrocyte node population (SONATA Nodes)", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--synaptic-data", help="Path to HDF5 with synapse positions", required=True)
@click.option("--morph-dir", help="Path to morphology folder", required=True)
@click.option("--parallel", help="Parallelize with 'multiprocessing'", is_flag=True, default=False)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
def neuroglial_finalize(
        input_file, output_file, astrocytes, microdomains, synaptic_data, morph_dir, parallel, seed):
    """For each astrocyte-neuron connection annotate the closest morphology section, segment, offset
    for each synapse.

    It adds in neuroglial connectivity the following properties:
        - astrocyte_section_id: int32
        - astrocyte_segment_id: int32
        - astrocyte_segment_offset: float32
    """
    import shutil
    from archngv.core.datasets import CellData, NeuroglialConnectivity
    from archngv.building.exporters.edge_populations import add_properties_to_edge_population
    from archngv.app.utils import apply_parallel_function

    paths = {
        'microdomains': microdomains,
        'synaptic_data': synaptic_data,
        'neuroglial_connectivity': input_file,
        'morph_dir': morph_dir
    }

    ng_connectivity = NeuroglialConnectivity(input_file)

    properties = _neuroglial_properties(
        seed=seed,
        astrocytes=CellData(astrocytes),
        n_connections=len(ng_connectivity),
        paths=paths,
        map_func=apply_parallel_function if parallel else map
    )

    shutil.copyfile(input_file, output_file)

    # add the new properties to the copied out file
    add_properties_to_edge_population(output_file, ng_connectivity.name, properties)


@click.command(name="glialglial-connectivity")
@click.option("--astrocytes", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--touches-dir", help="Path to touches directory", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--output-connectivity", help="Path to output HDF5 (connectivity)", required=True)
def build_glialglial_connectivity(astrocytes, touches_dir, seed, output_connectivity):
    """Generate connectivitiy betwen astrocytes (G-G)"""
    # pylint: disable=redefined-argument-from-local,too-many-locals
    from archngv.core.datasets import CellData
    from archngv.building.connectivity.glialglial import generate_glialglial
    from archngv.building.exporters.edge_populations import glialglial_connectivity

    from archngv.app.logger import LOGGER

    LOGGER.info('Seed: %d', seed)
    numpy.random.seed(seed)

    LOGGER.info('Creating symmetric connections from touches...')
    glialglial_data = generate_glialglial(touches_dir)

    LOGGER.info('Exporting to SONATA file...')
    glialglial_connectivity(glialglial_data, len(CellData(astrocytes)), output_connectivity)

    LOGGER.info("Done!")


@click.command(name="endfeet-area")
@click.option("--config", help="Path to YAML config", required=True)
@click.option("--vasculature-mesh", help="Path to vasculature mesh", required=True)
@click.option("--gliovascular-connectivity", help="Path to sonata gliovascular file", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output file (HDF5)", required=True)
def build_endfeet_surface_meshes(config, vasculature_mesh, gliovascular_connectivity, seed, output):
    """Generate the astrocytic endfeet geometries on the surface of the vasculature mesh.
    """
    import openmesh

    from archngv.core.datasets import GliovascularConnectivity
    from archngv.building.endfeet_reconstruction.area_generation import endfeet_area_generation
    from archngv.building.exporters.export_endfeet_areas import export_endfeet_areas

    from archngv.app.logger import LOGGER

    numpy.random.seed(seed)
    LOGGER.info('Seed: %d', seed)

    config = load_yaml(config)

    LOGGER.info('Load vasculature mesh at %s', vasculature_mesh)
    vasculature_mesh = openmesh.read_trimesh(vasculature_mesh)

    gliovascular_connectivity = GliovascularConnectivity(gliovascular_connectivity)
    endfeet_points = gliovascular_connectivity.vasculature_surface_targets()

    LOGGER.info('Setting up generator...')
    data_generator = endfeet_area_generation(
        vasculature_mesh=vasculature_mesh,
        parameters=config,
        endfeet_points=endfeet_points
    )

    LOGGER.info("Export to HDF5...")
    export_endfeet_areas(output, data_generator, len(endfeet_points))

    LOGGER.info('Done!')


def _synthesize(astrocyte_index, seed, paths, config):
    # imports must be local, otherwise when used with modules, they use numpy of the loaded
    # module which might be outdated
    from archngv.building.morphology_synthesis.data_extraction import astrocyte_circuit_data
    from archngv.building.morphology_synthesis.full_astrocyte import synthesize_astrocyte
    from archngv.app.utils import random_generator

    seed = hash((seed, astrocyte_index)) % (2 ** 32)
    rng = random_generator(seed)

    morph = synthesize_astrocyte(astrocyte_index, paths, config, rng)
    cell_properties = astrocyte_circuit_data(astrocyte_index, paths, rng)[0]
    morph.write(Path(paths.morphology_directory, cell_properties.name[0] + '.h5'))


@click.command()
@click.option("--config", help="Path to synthesis YAML config", required=True)
@click.option("--tns-distributions", help="Path to TNS distributions (JSON)", required=True)
@click.option("--tns-parameters", help="Path to TNS parameters (JSON)", required=True)
@click.option("--tns-context", help="Path to TNS context (JSON)", required=True)
@click.option("--er-data", help="Path to the Endoplasmic Reticulum data (JSON)", required=True)
@click.option("--astrocytes", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option(
    "--gliovascular-connectivity", help="Path to gliovascular connectivity sonata", required=True)
@click.option(
    "--neuroglial-connectivity", help="Path to neuroglial connectivity (HDF5)", required=True)
@click.option("--endfeet-areas", help="Path to HDF5 endfeet areas", required=True)
@click.option("--neuronal-connectivity", help="Path to HDF5 with synapse positions", required=True)
@click.option("--out-morph-dir", help="Path to output morphology folder", required=True)
@click.option("--parallel", help="Use Dask's mpi client", is_flag=True, default=False)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
def synthesis(config,
        tns_distributions,
        tns_parameters,
        tns_context,
        er_data,
        astrocytes,
        microdomains,
        gliovascular_connectivity,
        neuroglial_connectivity,
        endfeet_areas,
        neuronal_connectivity,
        out_morph_dir,
        parallel,
        seed):
    # pylint: disable=too-many-arguments
    """Cli interface to synthesis."""
    import time

    from dask import bag
    from dask.distributed import Client, progress
    import dask_mpi
    from archngv.core.datasets import CellData
    from archngv.building.morphology_synthesis.data_structures import SynthesisInputPaths

    if parallel:
        dask_mpi.initialize()
        client = Client()
    else:
        client = Client(processes=False, threads_per_worker=1)

    Path(out_morph_dir).mkdir(exist_ok=True, parents=True)
    config = load_yaml(config)
    n_astrocytes = len(CellData(astrocytes))
    paths = SynthesisInputPaths(
        astrocytes=astrocytes,
        microdomains=microdomains,
        neuronal_connectivity=neuronal_connectivity,
        gliovascular_connectivity=gliovascular_connectivity,
        neuroglial_connectivity=neuroglial_connectivity,
        endfeet_areas=endfeet_areas,
        tns_parameters=tns_parameters,
        tns_distributions=tns_distributions,
        tns_context=tns_context,
        er_data=er_data,
        morphology_directory=out_morph_dir)

    synthesize = bag.from_sequence(range(n_astrocytes), partition_size=1) \
        .map(_synthesize, seed=seed, paths=paths, config=config) \
        .persist()
    # print is intentional here because it is for showing the progress bar title
    print(f'Synthesizing {n_astrocytes} astrocytes')
    progress(synthesize)
    synthesize.compute()

    time.sleep(10)  # this sleep is necessary to let dask syncronize state across the cluster
    client.retire_workers()
