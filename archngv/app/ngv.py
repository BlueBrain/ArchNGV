"""ngv cli"""
# pylint: disable=too-many-statements
from pathlib import Path

import click
import numpy
import voxcell

from archngv.app.logger import LOGGER
from archngv.app.utils import load_yaml, write_json


@click.command()
@click.option("--bioname", help="Path to bioname folder", required=True)
@click.option("-o", "--output", help="Path to output file (JSON)", required=True)
def ngv_config(bioname, output):
    """Build NGV SONATA config"""
    from archngv.building.config import build_ngv_config

    bioname = Path(bioname).resolve()
    manifest = load_yaml(Path(bioname, "MANIFEST.yaml"))

    # resolving relative paths is needed for the cases where the executor is not
    # at the same place as the data. For example, pytest creates an isolated filesystem
    # and changes the current directory to that. Relative paths without resolution
    # would be resolved with respect to that isolated filesystem, producing invalid
    # paths.
    for key, value in manifest["common"].items():
        if isinstance(value, str) and not Path(value).is_absolute():
            manifest["common"][key] = str(Path(bioname, value).resolve())

    write_json(filepath=output, data=build_ngv_config(root_dir=bioname, manifest=manifest))


# pylint: disable=redefined-builtin
@click.command()
@click.option("-i", "--input", help="Path to input SONATA Nodes HDF5", required=True)
@click.option("--hoc", help="HOC template file name", required=True)
@click.option("-o", "--output", help="Path to output SONATA Nodes HDF5", required=True)
def assign_emodels(input, hoc, output):
    """Assign `model_template` attribute to node population"""
    emodels = voxcell.CellCollection.load_sonata(input)
    cols = list(emodels.properties)
    emodels.properties["model_template"] = f"hoc:{hoc}"
    emodels.properties = emodels.properties.drop(columns=cols)
    emodels.save_sonata(output)


@click.command()
@click.option("--config", help="Path to astrocyte placement YAML config", required=True)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option("--vasculature", help="Path to vasculature node population", required=True)
@click.option(
    "--seed",
    help="Pseudo-random generator seed",
    type=int,
    default=0,
    show_default=True,
)
@click.option("-o", "--output", help="Path to output SONATA nodes file", required=True)
def cell_placement(config, atlas, atlas_cache, vasculature, seed, output):
    """
    Generate astrocyte positions and radii inside the bounding box of the vasculature
    dataset.

    Astrocytes are placed without colliding with the vasculature geometry or with other
    astrocytic somata.
    """
    # pylint: disable=too-many-locals

    from spatial_index import SphereIndex
    from vascpy import PointVasculature
    from voxcell.nexus.voxelbrain import Atlas

    from archngv.building.cell_placement.positions import create_positions
    from archngv.building.checks import assert_bbox_alignment
    from archngv.building.exporters.node_populations import export_astrocyte_population
    from archngv.spatial import BoundingBox

    numpy.random.seed(seed)
    LOGGER.info("Seed: %d", seed)

    config = load_yaml(config)["cell_placement"]

    atlas = Atlas.open(atlas, cache_dir=atlas_cache)
    voxelized_intensity = atlas.load_data(config["density"])
    voxelized_bnregions = atlas.load_data("brain_regions")

    assert numpy.issubdtype(voxelized_intensity.raw.dtype, numpy.floating)

    spatial_indexes = []
    if vasculature is not None:

        vasc = PointVasculature.load_sonata(vasculature)

        assert_bbox_alignment(
            BoundingBox.from_points(vasc.points),
            BoundingBox(voxelized_intensity.bbox[0], voxelized_intensity.bbox[1]),
        )

        spatial_indexes.append(SphereIndex(vasc.points, 0.5 * vasc.diameters))

    LOGGER.info("Generating cell positions / radii...")
    somata_positions, somata_radii = create_positions(
        config,
        voxelized_intensity,
        voxelized_bnregions,
        spatial_indexes=spatial_indexes,
    )

    cell_names = ["GLIA_{:013d}".format(index) for index in range(len(somata_positions))]

    LOGGER.info("Export to CellData...")
    export_astrocyte_population(
        output, cell_names, somata_positions, somata_radii, mtype="ASTROCYTE"
    )

    LOGGER.info("Done!")


@click.command()
@click.option("--somata-file", help="Path to sonata somata file", required=True)
@click.option("--emodels-file", help="Path to sonata emodels file", required=True)
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def finalize_astrocytes(somata_file, emodels_file, output):
    """Build the finalized astrocyte node population by merging the different astrocyte
    properties"""
    from archngv.core.constants import Population

    somata = voxcell.CellCollection.load(somata_file)
    emodels = voxcell.CellCollection.load(emodels_file)
    somata.properties["model_template"] = emodels.properties["model_template"]
    somata.population_name = Population.ASTROCYTES
    somata.save_sonata(output)


@click.command()
@click.option("--config", help="Path to astrocyte microdomains YAML config", required=True)
@click.option(
    "--astrocytes",
    help="Path to the sonata file with astrocyte's positions and radii",
    required=True,
)
@click.option("--atlas", help="Atlas URL / path", required=True)
@click.option("--atlas-cache", help="Path to atlas cache folder", default=None, show_default=True)
@click.option(
    "--seed",
    help="Pseudo-random generator seed",
    type=int,
    default=0,
    show_default=True,
)
@click.option("-o", "--output-file-path", help="Path to output hdf5 file", required=True)
def build_microdomains(config, astrocytes, atlas, atlas_cache, seed, output_file_path):
    """Generate astrocyte microdomain tessellation as a partition of space into convex
    polygons."""
    # pylint: disable=too-many-locals
    from scipy import stats
    from voxcell.nexus.voxelbrain import Atlas

    from archngv.building.exporters.export_microdomains import export_microdomains
    from archngv.building.microdomains import (
        generate_microdomain_tessellation,
        scaling_factor_from_overlap,
    )
    from archngv.spatial import BoundingBox

    LOGGER.info("Seed: %d", seed)
    numpy.random.seed(seed)

    config = load_yaml(config)["microdomains"]

    atlas = Atlas.open(atlas, cache_dir=atlas_cache)
    bbox_ranges = atlas.load_data("brain_regions").bbox

    astrocytes = voxcell.CellCollection.load_sonata(astrocytes)

    LOGGER.info("Generating microdomains...")
    microdomains = generate_microdomain_tessellation(
        generator_points=astrocytes.positions,
        generator_radii=astrocytes.properties["radius"].to_numpy(),
        bounding_box=BoundingBox(bbox_ranges[0], bbox_ranges[1]),
    )

    LOGGER.info("Generating overlapping microdomains...")
    overlap_distr = config["overlap_distribution"]["values"]
    overlap_distribution = stats.norm(loc=overlap_distr[0], scale=overlap_distr[1])

    scaling_factors = numpy.fromiter(
        map(
            scaling_factor_from_overlap,
            overlap_distribution.rvs(size=len(astrocytes)),
        ),
        dtype=float,
        count=len(astrocytes),
    )

    overlapping_microdomains = (
        dom.scale(factor) for dom, factor in zip(microdomains, scaling_factors)
    )

    LOGGER.info("Export overlapping microdomains...")
    export_microdomains(output_file_path, overlapping_microdomains, scaling_factors)

    LOGGER.info("Done!")


@click.command()
@click.option("--config", help="Path to astrocyte placement YAML config", required=True)
@click.option(
    "--astrocytes",
    help="Path to the sonata file with astrocyte's positions",
    required=True,
)
@click.option("--microdomains", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--vasculature", help="Path to vasculature sonata dataset", required=True)
@click.option(
    "--seed",
    help="Pseudo-random generator seed",
    type=int,
    default=0,
    show_default=True,
)
@click.option("--output", help="Path to output edges HDF5 (data)", required=True)
def gliovascular_connectivity(config, astrocytes, microdomains, vasculature, seed, output):
    """
    Build connectivity between astrocytes and the vasculature graph.
    """
    # pylint: disable=too-many-locals
    from vascpy import PointVasculature

    from archngv.building.connectivity.gliovascular import generate_gliovascular
    from archngv.building.exporters.edge_populations import write_gliovascular_connectivity
    from archngv.core.datasets import Microdomains

    LOGGER.info("Seed: %d", seed)
    numpy.random.seed(seed)

    astrocytes = voxcell.CellCollection.load_sonata(astrocytes)

    LOGGER.info("Generating gliovascular connectivity...")
    (
        endfoot_surface_positions,
        endfeet_to_astrocyte_mapping,
        endfeet_to_vasculature_mapping,
    ) = generate_gliovascular(
        cell_ids=numpy.arange(len(astrocytes), dtype=numpy.int64),
        astrocytic_positions=astrocytes.positions,
        astrocytic_domains=Microdomains(microdomains),
        vasculature=PointVasculature.load_sonata(vasculature),
        params=load_yaml(config)["gliovascular_connectivity"],
    )

    LOGGER.info("Exporting sonata edges...")
    write_gliovascular_connectivity(
        output,
        astrocytes,
        voxcell.CellCollection.load_sonata(vasculature),
        endfeet_to_astrocyte_mapping,
        endfeet_to_vasculature_mapping,
        endfoot_surface_positions,
    )

    LOGGER.info("Done!")


@click.command()
@click.option("--input-file", help="Path to sonata edges file (HDF5)", required=True)
@click.option("--output-file", help="Path to sonata edges file (HDF5)", required=True)
@click.option("--astrocytes", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--endfeet-areas", help="Path to HDF5 endfeet areas", required=True)
@click.option("--vasculature-sonata", help="Path to nodes for vasculature (HDF5)", required=True)
@click.option("--morph-dir", help="Path to morphology folder", required=True)
@click.option(
    "--seed",
    help="Pseudo-random generator seed",
    type=int,
    default=0,
    show_default=True,
)
@click.option("--parallel", help="Parallelize with 'multiprocessing'", is_flag=True)
def attach_endfeet_info_to_gliovascular_connectivity(
    input_file,
    output_file,
    astrocytes,
    endfeet_areas,
    vasculature_sonata,
    morph_dir,
    seed,
    parallel,
):
    """
    Finalizes gliovascular connectivity. It needs to be ran after synthesis and endfeet
    area growing. It copies to the input GliovascularConnectivity population and adds
    the following edge properties:

        - astrocyte_section_id
            The last section id of the astrocytic morphology that connects to the
            endfoot location on the vascular surface.

        - endfoot_compartment_length
            The extent of the endfoot across the medial axis of the segment that is
            located.

        - endfoot_compartment_diameter
            A diameter value so that diameter * length = volume of endfoot.

        - endfoot_compartment_perimeter
            A perimeter value so that perimeter * length = area of endfoot.
    """
    import shutil

    from vascpy import PointVasculature

    from archngv.app.utils import apply_parallel_function
    from archngv.building.endfeet_reconstruction.gliovascular_properties import (
        endfeet_mesh_properties,
    )
    from archngv.building.exporters.edge_populations import add_properties_to_edge_population
    from archngv.core.datasets import CellData, EndfootSurfaceMeshes, GliovascularConnectivity

    gv_connectivity = GliovascularConnectivity(input_file)

    # Retrieve mesh related endfeet properties to attach to the gliovascular edge
    # population.
    properties = endfeet_mesh_properties(
        seed=seed,
        astrocytes=CellData(astrocytes),
        gliovascular_connectivity=gv_connectivity,
        vasculature=PointVasculature.load_sonata(vasculature_sonata),
        endfeet_meshes=EndfootSurfaceMeshes(endfeet_areas),
        morph_dir=morph_dir,
        map_function=apply_parallel_function if parallel else map,
    )
    # copy gv file to the output location
    shutil.copyfile(input_file, output_file)

    # add the new properties to the copied out file
    add_properties_to_edge_population(output_file, gv_connectivity.name, properties)


@click.command()
@click.option(
    "--neurons-path",
    help="Path to neuron node population (SONATA Nodes)",
    required=True,
)
@click.option(
    "--astrocytes-path",
    help="Path to astrocyte node population (SONATA Nodes)",
    required=True,
)
@click.option("--microdomains-path", help="Path to microdomains structure (HDF5)", required=True)
@click.option(
    "--neuronal-connectivity-path",
    help="Path to neuron-neuron sonata edge file",
    required=True,
)
@click.option(
    "--seed",
    help="Pseudo-random generator seed",
    type=int,
    default=0,
    show_default=True,
)
@click.option("-o", "--output-path", help="Path to output file (SONATA Edges HDF5)", required=True)
def neuroglial_connectivity(
    neurons_path,
    astrocytes_path,
    microdomains_path,
    neuronal_connectivity_path,
    seed,
    output_path,
):
    """Generate connectivity between neurons (N) and astrocytes (G)"""
    # pylint: disable=too-many-locals

    from archngv.building.connectivity.neuroglial_generation import generate_neuroglial
    from archngv.building.exporters.edge_populations import write_neuroglial_connectivity
    from archngv.core.datasets import Microdomains, NeuronalConnectivity

    numpy.random.seed(seed)

    astrocytes_data = voxcell.CellCollection.load(astrocytes_path)

    LOGGER.info("Generating neuroglial connectivity...")

    data_iterator = generate_neuroglial(
        astrocytes=astrocytes_data,
        microdomains=Microdomains(microdomains_path),
        neuronal_connectivity=NeuronalConnectivity(neuronal_connectivity_path),
    )

    LOGGER.info("Exporting the per astrocyte files...")
    write_neuroglial_connectivity(
        data_iterator,
        neurons=voxcell.CellCollection.load(neurons_path),
        astrocytes=astrocytes_data,
        output_path=output_path,
    )

    LOGGER.info("Done!")


@click.command()
@click.option("--input-file-path", help="Path to input file (SONATA Edges HDF5)", required=True)
@click.option("--output-file-path", help="Path to output file (SONATA Edges HDF5)", required=True)
@click.option(
    "--astrocytes-path",
    help="Path to astrocyte node population (SONATA Nodes)",
    required=True,
)
@click.option("--microdomains-path", help="Path to microdomains structure (HDF5)", required=True)
@click.option("--synaptic-data-path", help="Path to HDF5 with synapse positions", required=True)
@click.option("--morph-dir", help="Path to morphology folder", required=True)
@click.option("--parallel", help="Parallelize with 'multiprocessing'", is_flag=True)
@click.option(
    "--seed",
    help="Pseudo-random generator seed",
    type=int,
    default=0,
    show_default=True,
)
def attach_morphology_info_to_neuroglial_connectivity(
    input_file_path,
    output_file_path,
    astrocytes_path,
    microdomains_path,
    synaptic_data_path,
    morph_dir,
    parallel,
    seed,
):
    """For each astrocyte-neuron connection annotate the closest morphology section,
    segment, offset for each synapse.

    It adds in neuroglial connectivity the following properties:
        - astrocyte_section_id: int32
        - astrocyte_segment_id: int32
        - astrocyte_segment_offset: float32
    """
    import shutil

    from archngv.app.utils import apply_parallel_function
    from archngv.building.exporters.edge_populations import add_properties_to_edge_population
    from archngv.building.morphology_synthesis.neuroglial_properties import (
        astrocyte_morphology_properties,
    )
    from archngv.core.datasets import CellData, NeuroglialConnectivity

    paths = {
        "microdomains": microdomains_path,
        "synaptic_data": synaptic_data_path,
        "neuroglial_connectivity": input_file_path,
        "morph_dir": morph_dir,
    }

    ng_connectivity = NeuroglialConnectivity(input_file_path)

    properties = astrocyte_morphology_properties(
        seed=seed,
        astrocytes=CellData(astrocytes_path),
        n_connections=len(ng_connectivity),
        paths=paths,
        map_function=apply_parallel_function if parallel else map,
    )

    # make a copy of the original and modify
    shutil.copyfile(input_file_path, output_file_path)

    # add the new properties to the copied out file
    add_properties_to_edge_population(output_file_path, ng_connectivity.name, properties)


@click.command(name="glialglial-connectivity")
@click.option("--astrocytes", help="Path to HDF5 with somata positions and radii", required=True)
@click.option("--touches-dir", help="Path to touches directory", required=True)
@click.option(
    "--seed",
    help="Pseudo-random generator seed",
    type=int,
    default=0,
    show_default=True,
)
@click.option("--output-connectivity", help="Path to output HDF5 (connectivity)", required=True)
def build_glialglial_connectivity(astrocytes, touches_dir, seed, output_connectivity):
    """Generate connectivitiy between astrocytes (G-G)"""
    # pylint: disable=redefined-argument-from-local,too-many-locals
    from archngv.building.connectivity.glialglial import generate_glialglial
    from archngv.building.exporters.edge_populations import write_glialglial_connectivity
    from archngv.core.datasets import CellData

    LOGGER.info("Seed: %d", seed)
    numpy.random.seed(seed)

    LOGGER.info("Creating symmetric connections from touches...")
    glialglial_data = generate_glialglial(touches_dir)

    LOGGER.info("Exporting to SONATA file...")
    write_glialglial_connectivity(glialglial_data, len(CellData(astrocytes)), output_connectivity)

    LOGGER.info("Done!")


@click.command(name="endfeet-area")
@click.option("--config-path", help="Path to YAML config", required=True)
@click.option("--vasculature-mesh-path", help="Path to vasculature mesh", required=True)
@click.option(
    "--gliovascular-connectivity-path",
    help="Path to sonata gliovascular file",
    required=True,
)
@click.option(
    "--seed",
    help="Pseudo-random generator seed",
    type=int,
    default=0,
    show_default=True,
)
@click.option("-o", "--output-path", help="Path to output file (HDF5)", required=True)
def build_endfeet_surface_meshes(
    config_path,
    vasculature_mesh_path,
    gliovascular_connectivity_path,
    seed,
    output_path,
):
    """Generate the astrocytic endfeet geometries on the surface of the vasculature
    mesh."""
    import openmesh

    from archngv.building.endfeet_reconstruction.area_generation import endfeet_area_generation
    from archngv.building.exporters.export_endfeet_areas import export_endfeet_meshes
    from archngv.core.datasets import GliovascularConnectivity

    numpy.random.seed(seed)
    LOGGER.info("Seed: %d", seed)

    config = load_yaml(config_path)["endfeet_surface_meshes"]

    LOGGER.info("Load vasculature mesh at %s", vasculature_mesh_path)
    vasculature_mesh = openmesh.read_trimesh(vasculature_mesh_path)

    endfeet_points = GliovascularConnectivity(
        gliovascular_connectivity_path
    ).vasculature_surface_targets()

    LOGGER.info("Setting up generator...")
    data_generator = endfeet_area_generation(
        vasculature_mesh=vasculature_mesh,
        parameters=config,
        endfeet_points=endfeet_points,
    )

    LOGGER.info("Export to HDF5...")
    export_endfeet_meshes(output_path, data_generator, len(endfeet_points))

    LOGGER.info("Done!")


def _synthesize(astrocyte_index, seed, paths, config):
    # imports must be local, otherwise when used with modules, they use numpy of the loaded
    # module which might be outdated
    from archngv.app.utils import random_generator
    from archngv.building.morphology_synthesis.data_extraction import astrocyte_circuit_data
    from archngv.building.morphology_synthesis.full_astrocyte import synthesize_astrocyte

    seed = hash((seed, astrocyte_index)) % (2**32)
    rng = random_generator(seed)

    morph = synthesize_astrocyte(astrocyte_index, paths, config, rng)
    cell_properties = astrocyte_circuit_data(astrocyte_index, paths, rng)[0]
    morph.write(Path(paths.morphology_directory, cell_properties.name[0] + ".h5"))


@click.command()
@click.option("--config-path", help="Path to synthesis YAML config", required=True)
@click.option("--tns-distributions-path", help="Path to TNS distributions (JSON)", required=True)
@click.option("--tns-parameters-path", help="Path to TNS parameters (JSON)", required=True)
@click.option("--tns-context-path", help="Path to TNS context (JSON)", required=True)
@click.option(
    "--er-data-path",
    help="Path to the Endoplasmic Reticulum data (JSON)",
    required=True,
)
@click.option(
    "--astrocytes-path",
    help="Path to HDF5 with somata positions and radii",
    required=True,
)
@click.option("--microdomains-path", help="Path to microdomains structure (HDF5)", required=True)
@click.option(
    "--gliovascular-connectivity-path",
    help="Path to gliovascular connectivity sonata",
    required=True,
)
@click.option(
    "--neuroglial-connectivity-path",
    help="Path to neuroglial connectivity (HDF5)",
    required=True,
)
@click.option("--endfeet-areas-path", help="Path to HDF5 endfeet areas", required=True)
@click.option(
    "--neuronal-connectivity-path",
    help="Path to HDF5 with synapse positions",
    required=True,
)
@click.option("--out-morph-dir", help="Path to output morphology folder", required=True)
@click.option("--parallel", help="Use Dask's mpi client", is_flag=True)
@click.option(
    "--seed",
    help="Pseudo-random generator seed",
    type=int,
    default=0,
    show_default=True,
)
def synthesis(
    config_path,
    tns_distributions_path,
    tns_parameters_path,
    tns_context_path,
    er_data_path,
    astrocytes_path,
    microdomains_path,
    gliovascular_connectivity_path,
    neuroglial_connectivity_path,
    endfeet_areas_path,
    neuronal_connectivity_path,
    out_morph_dir,
    parallel,
    seed,
):
    # pylint: disable=too-many-arguments
    """Cli interface to synthesis."""
    import time

    import dask_mpi
    from dask import bag
    from dask.distributed import Client, progress

    from archngv.building.morphology_synthesis.data_structures import SynthesisInputPaths
    from archngv.core.datasets import CellData

    if parallel:
        dask_mpi.initialize()
        client = Client()
    else:
        client = Client(processes=False, threads_per_worker=1)

    Path(out_morph_dir).mkdir(exist_ok=True, parents=True)
    config = load_yaml(config_path)["synthesis"]
    n_astrocytes = len(CellData(astrocytes_path))
    paths = SynthesisInputPaths(
        astrocytes=astrocytes_path,
        microdomains=microdomains_path,
        neuronal_connectivity=neuronal_connectivity_path,
        gliovascular_connectivity=gliovascular_connectivity_path,
        neuroglial_connectivity=neuroglial_connectivity_path,
        endfeet_areas=endfeet_areas_path,
        tns_parameters=tns_parameters_path,
        tns_distributions=tns_distributions_path,
        tns_context=tns_context_path,
        er_data=er_data_path,
        morphology_directory=out_morph_dir,
    )

    synthesize = (
        bag.from_sequence(range(n_astrocytes), partition_size=1)
        .map(_synthesize, seed=seed, paths=paths, config=config)
        .persist()
    )
    # print is intentional here because it is for showing the progress bar title
    print(f"Synthesizing {n_astrocytes} astrocytes")
    progress(synthesize)
    synthesize.compute()

    time.sleep(10)  # this sleep is necessary to let dask synchronize state across the cluster
    client.retire_workers()
