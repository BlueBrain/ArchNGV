"""
Generate gliovascular (G-V) connectivity
"""
import os
import click
from archngv.exceptions import NGVError


@click.group(help=__doc__)
def group():
    # pylint: disable=missing-docstring
    pass


@group.command()
@click.option("--config", help="Path to astrocyte placement YAML config", required=True)
@click.option("--astrocytes", help="Path to the sonata file with astrocyte's positions", required=True)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--vasculature", help="Path to vasculature dataset", required=True)
@click.option("--vasculature-sonata", help="Path to vasculature sonata dataset", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--output", help="Path to output edges HDF5 (data)", required=True)
def connectivity(config, astrocytes, microdomains, vasculature, vasculature_sonata, seed, output):
    # pylint: disable=missing-docstring,redefined-argument-from-local,too-many-locals
    import numpy as np
    from voxcell import CellCollection

    from archngv.core.datasets import (
        Vasculature,
        MicrodomainTesselation
    )

    from archngv.building.connectivity.gliovascular_generation import generate_gliovascular
    from archngv.building.exporters.edge_populations import gliovascular_connectivity

    from archngv.app.logger import LOGGER
    from archngv.app.utils import load_yaml

    LOGGER.info('Seed: %d', seed)
    np.random.seed(seed)

    params = load_yaml(config)
    vasculature = Vasculature.load(vasculature)

    LOGGER.info('Generating gliovascular connectivity...')

    astrocyte_positions = CellCollection.load_sonata(astrocytes).positions
    astrocyte_idx = np.arange(len(astrocyte_positions), dtype=np.int64)
    microdomains = MicrodomainTesselation(microdomains)

    (
        endfoot_surface_positions,
        endfeet_to_astrocyte_mapping,
        endfeet_to_vasculature_mapping
    ) = generate_gliovascular(astrocyte_idx, astrocyte_positions, microdomains, vasculature, params)

    LOGGER.info('Exporting sonata edges...')
    gliovascular_connectivity(
        output,
        CellCollection.load_sonata(astrocytes),
        CellCollection.load_sonata(vasculature_sonata),
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


class Worker:
    """Endfeet properties parallel helper"""
    def __init__(self, seed):
        self._seed = seed

    def __call__(self, data):
        """
        Args:
            data (dict)
        """
        import numpy as np
        np.random.seed(hash((self._seed, data['index'])) % (2 ** 32))
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


def _dispatch_data(astrocytes, gv_connectivity, vasculature, endfeet_meshes, morph_dir):
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
        morphology_path = os.path.join(morph_dir, morphology_name + '.h5')
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
    import numpy as np

    endfeet_ids = gv_connectivity.get_property('endfoot_id')
    n_endfeet = len(endfeet_ids)

    if not np.array_equal(endfeet_ids, np.arange(n_endfeet)):
        raise NGVError('endfeet_ids should be a contiguous array from 0 to number of endfeet')

    properties = {
        'astrocyte_section_id': np.empty(n_endfeet, dtype=np.uint32),
        'endfoot_compartment_length': np.empty(n_endfeet, dtype=np.float32),
        'endfoot_compartment_diameter': np.empty(n_endfeet, dtype=np.float32),
        'endfoot_compartment_perimeter': np.empty(n_endfeet, dtype=np.float32)
    }

    it_results = map_func(
        Worker(seed),
        _dispatch_data(astrocytes, gv_connectivity, vasculature, endfeet_meshes, morph_dir)
    )

    for ids, section_ids, lengths, diameters, perimeters in it_results:

        properties['astrocyte_section_id'][ids] = section_ids
        properties['endfoot_compartment_length'][ids] = lengths
        properties['endfoot_compartment_diameter'][ids] = diameters
        properties['endfoot_compartment_perimeter'][ids] = perimeters

    return properties


@group.command()
@click.option("--input-file", help="Path to sonata edges file (HDF5)", required=True)
@click.option("--output-file", help="Path to sonata edges file (HDF5)", required=True)
@click.option("--astrocytes", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--endfeet-areas", help="Path to HDF5 endfeet areas", required=True)
@click.option("--vasculature-sonata", help="Path to nodes for vasculature (HDF5)", required=True)
@click.option("--morph-dir", help="Path to morphology folder", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("--parallel", help="Parallelize with 'multiprocessing'", is_flag=True, default=False)
def finalize(input_file, output_file, astrocytes, endfeet_areas, vasculature_sonata, morph_dir, seed, parallel):
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
    from archngv.core.datasets import CellData, GliovascularConnectivity, Vasculature, EndfootSurfaceMeshes
    from archngv.building.exporters.edge_populations import add_properties_to_edge_population
    from archngv.app.utils import apply_parallel_function

    gv_connectivity = GliovascularConnectivity(input_file)

    properties = _endfeet_properties(
        seed=seed,
        astrocytes=CellData(astrocytes),
        gv_connectivity=gv_connectivity,
        vasculature=Vasculature.load_sonata(vasculature_sonata),
        endfeet_meshes=EndfootSurfaceMeshes(endfeet_areas),
        morph_dir=morph_dir,
        map_func=apply_parallel_function if parallel else map
    )

    # copy gv file to the output location
    shutil.copyfile(input_file, output_file)

    # add the new properties to the copied out file
    add_properties_to_edge_population(output_file, gv_connectivity.name, properties)
