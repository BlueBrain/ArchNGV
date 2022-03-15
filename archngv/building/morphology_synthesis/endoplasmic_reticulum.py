"""
Endoplasmic reticulum data creation and distribution methods
"""
import json

import morphio
import numpy as np
import pandas as pd
import tmd
from tmd.io.conversion import _section_to_data
from tmd.Topology.methods import _filtration_function
from tmd.Topology.persistent_properties import NoProperty
from tmd.Topology.statistics import get_lengths


def add_endoplasmic_reticulum_to_morphology(morphology, barcodes_file):
    """Adds endoplasmic reticulum placeholder values.

    Morphologies need to be populated with ER. However, the algos for synthesizing the ER
    is not yet in place. Thus, in order to enable the testing for the neurodamus implementations,
     we add placeholder values for the time being.

    Args:
        morhology (morphio.mut.Morphology): morphio mutable morphology
    """
    bio_barcodes = _load_barcodes(barcodes_file)
    syn_barcodes = _morphology_to_tree_barcodes(morphology)
    barcode_matching = _match_barcodes(syn_barcodes, bio_barcodes)
    section_ids_to_er = _assign_er_to_sections(syn_barcodes, bio_barcodes, barcode_matching)

    section_indices = np.sort(list(section_ids_to_er.keys()))

    surface_areas = np.empty(section_indices.size, dtype=np.float32)
    volumes = np.empty(section_indices.size, dtype=np.float32)
    filament_counts = np.empty(section_indices.size, dtype=np.int32)

    for i, sid in enumerate(section_indices):
        surface_areas[i], volumes[i], filament_counts[i] = section_ids_to_er[sid]

    er = morphology.endoplasmic_reticulum
    er.section_indices = section_indices
    er.surface_areas = surface_areas
    er.volumes = volumes
    er.filament_counts = filament_counts


def _morphology_to_tree_barcodes(morphology):
    class MorphIOSectionId:
        """Class for mapping morphio section id to tmd point tree"""

        def __init__(self, _):

            self.data = {}

        def update(self, section_id, termination_id):
            """Update termination with section id"""
            self.data[termination_id] = section_id

    def path_distances_from_terminations(neurite, bar_terminations):
        """Extract path distances from terminations"""
        return neurite.get_point_path_distances()[np.array(bar_terminations)]

    def extract_morphio_section_ids(neurite, terms):
        """Extract section ids from neurite"""
        section_ids = neurite.properties["morphio_sid"]
        return [section_ids[term] for term in terms]

    if not isinstance(morphology, morphio.Morphology):
        morphology = morphio.Morphology(morphology)

    syn_neurites = list(
        _convert_morphio_trees(morphology, property_builders={"morphio_sid": MorphIOSectionId})
    )

    apply_classes = {
        "morphio_sid": extract_morphio_section_ids,
        "path_distances_from_terminations": path_distances_from_terminations,
    }

    return [_get_barcode(syn_neurite, apply=apply_classes) for syn_neurite in syn_neurites]


def _convert_morphio_trees(cell, property_builders=None):
    """Convert morphio morphology's trees to tmd ones

    Args:
        cell (Union[morphio.Morphology, morphio.mut.Morphology]):
            morphio neuron object

    Yields:
        tuple:
            - tree (Tree): The constructed tmd Tree
            - tree_types (dict): The neuron tree types
    """

    class Tree(tmd.Tree.Tree):
        """Inherit from tree to add properties"""

        def __init__(self, x, y, z, d, t, p, properties=None):
            super().__init__(x, y, z, d, t, p)
            self.properties = properties

    if property_builders is None:
        property_builders = {}

    total_points = len(cell.diameters)

    x = np.empty(total_points, dtype=np.float32)
    y = np.empty(total_points, dtype=np.float32)
    z = np.empty(total_points, dtype=np.float32)
    d = np.empty(total_points, dtype=np.float32)
    t = np.empty(total_points, dtype=np.int32)
    p = np.empty(total_points, dtype=np.int64)

    section_final_nodes = np.empty(total_points, dtype=np.int32)

    tree_end = 0
    for root in cell.root_sections:

        tree_length = 0
        tree_beg = tree_end

        p_builders = {name: builder(cell) for name, builder in property_builders.items()}

        for section in root.iter():

            # root sections have parent -1
            if section.is_root:
                start = 0
                parent = -1
            else:
                # tmd does not use a duplicate point representation
                # thus the first point of the section is dropped
                start = 1
                parent = section_final_nodes[section.parent.id]

            n, data = _section_to_data(section, tree_length, start, parent)

            x[tree_end : n + tree_end] = data.points[:, 0]
            y[tree_end : n + tree_end] = data.points[:, 1]
            z[tree_end : n + tree_end] = data.points[:, 2]
            d[tree_end : n + tree_end] = data.diameters
            t[tree_end : n + tree_end] = data.section_type
            p[tree_end : n + tree_end] = data.parents

            tree_end += n
            tree_length += n

            # keep track of the last node in the section because we need
            # to establish the correct connectivity when we omit the first
            # point from the children sections
            section_final_nodes[section.id] = tree_length - 1

            for builder in p_builders.values():
                builder.update(section.id, section_final_nodes[section.id])

        yield Tree(
            x=x[tree_beg:tree_end],
            y=y[tree_beg:tree_end],
            z=z[tree_beg:tree_end],
            d=d[tree_beg:tree_end],
            t=t[tree_beg:tree_end],
            p=p[tree_beg:tree_end],
            properties={name: builder.data for name, builder in p_builders.items()},
        )


def _get_barcode(neurite, apply=None):
    """Get barcode and barcode data from neurite. If apply is empty, barcode_data will be empty"""
    barcode, bars_to_terms = _tree_to_property_barcode(
        tree=neurite, filtration_function=_filtration_function("path_distances")
    )

    if apply is None:
        apply = {}

    barcode_data = [
        {key: list(func(neurite, terms)) for key, func in apply.items()} for terms in bars_to_terms
    ]

    return barcode, barcode_data


def _tree_to_property_barcode(tree, filtration_function, property_class=NoProperty):
    """Decompose a tree data structure into a barcode, where each bar in the barcode
    is optionally linked with a property determined by property_class.

    Args:

        filtration_function (Callable[tree] -> np.ndarray):
            The filtration function to apply on the tree

        property_class (PersistentProperty, optional): A PersistentProperty class.By
            default the NoProperty is used which does not add entries in the barcode.

    Returns:
        barcode (list): A list of bars [bar1, bar2, ..., barN], where each bar is a
            list of:
                - filtration value start
                - filtration value end
                - property_value1
                - property_value2
                - ...
                - property_valueN
        bars_to_points: A list of point ids for each bar in the barcode. Each list
            corresponds to the set of endpoints (i.e. the end point of each section)
            that belong to the corresponding persistent component - or bar.
    """
    point_values = filtration_function(tree)

    beg, _ = tree.sections
    parents, children = tree.parents_children

    prop = property_class(tree)

    active = tree.get_bif_term() == 0
    alives = np.where(active)[0]
    point_ids_track = {al: [al] for al in alives}
    bars_to_points = []

    ph = []
    while len(alives) > 1:
        for alive in alives:

            p = parents[alive]
            c = children[p]

            if np.alltrue(active[c]):

                active[p] = True
                active[c] = False

                mx = np.argmax(abs(point_values[c]))
                mx_id = c[mx]

                c = np.delete(c, mx)

                for ci in c:
                    component_id = np.where(beg == p)[0][0]
                    ph.append([point_values[ci], point_values[p]] + prop.get(component_id))
                    bars_to_points.append(point_ids_track[ci])

                point_values[p] = point_values[mx_id]
                point_ids_track[p] = point_ids_track[mx_id] + [p]
        alives = np.where(active)[0]

    ph.append(
        [point_values[alives[0]], 0] + prop.infinite_component(beg[0])
    )  # Add the last alive component
    bars_to_points.append(point_ids_track[alives[0]])

    return ph, bars_to_points


def create_bio_barcodes(morphology_paths):
    """Create biological barcodes from a list of morphology paths"""
    bio_neurites = []

    for bio_path in morphology_paths:
        cell = morphio.Morphology(bio_path)
        neurites = _convert_morphio_trees(cell, property_builders={"ER": EndoplasmicReticulum})
        bio_neurites.extend(neurites)

    apply_cls = {"ER": _er_from_bar_terminations}
    return [_get_barcode(bio_neurite, apply=apply_cls) for bio_neurite in bio_neurites]


class EndoplasmicReticulum:
    """
    Args:
        morphology (morphio.Morphology):
    Attrs:
        data (dict): A mapping between termination_id and morphology section data
    """

    def __init__(self, morphology):

        self._morphology = morphology
        er_data = self.extract_morphio_er_data(morphology)

        self._section_id_to_er = dict(
            zip(er_data.index, er_data.itertuples(index=False, name="ER"))
        )

        self.data = {}

    @staticmethod
    def extract_morphio_er_data(morphology):
        """Extracts morphio endoplasmic reticulum info as a pandas dataframe

        Returns:
            pandas.DataFrame: The ER data in a dataframe having three columns
                corresponding to the surface area, volume and filament count of the
                sections that have ER in the morphology. The dataframe index stores the
                section index where the ER was located.
        """
        er = morphology.endoplasmic_reticulum

        section_indices = er.section_indices
        counts = er.filament_counts

        if len(counts) == 0:
            counts = np.zeros(len(section_indices), dtype=np.float32)

        return pd.DataFrame(
            {
                "surface_area": er.surface_areas,
                "volume": er.volumes,
                "filament_count": counts,
            },
            index=section_indices,
        )

    def update(self, section_id, termination_id):
        """Associates the tmd terminations with the morphio section ER data"""
        try:
            self.data[termination_id] = self._section_id_to_er[section_id]
        except KeyError:
            self.data[termination_id] = None


def _er_from_bar_terminations(neurite, bar_terminations):
    """Get er from bar terminations"""
    er_data = neurite.properties["ER"]
    path_distances = neurite.get_point_path_distances()
    return (
        [path_distances[term] for term in bar_terminations],
        [er_data[term] for term in bar_terminations],
    )


def _save_barcodes(filepath, barcodes):
    """Saves a list of barcodes"""
    with open(filepath, "w") as outfile:
        json.dump(barcodes, outfile, indent=2)


def _load_barcodes(filepath):
    """Loads a list of barcodes from a json file"""
    with open(filepath, "r") as fd:
        return json.load(fd)


def _match_barcodes(to_match, available_barcodes):
    """Find a match for each barcode based on its number of bars"""
    bars_per_barcode_match = np.fromiter((len(barcode[0]) for barcode in to_match), dtype=np.int32)
    bars_per_barcode_avail = np.fromiter(
        (len(barcode[0]) for barcode in available_barcodes), dtype=np.int32
    )

    # identical barcodes
    if len(bars_per_barcode_match) == len(bars_per_barcode_avail) and np.all(
        bars_per_barcode_match == bars_per_barcode_avail
    ):
        return np.arange(len(bars_per_barcode_match), dtype=np.int32)

    closest_bar = lambda n_bars: np.argmin(np.abs(bars_per_barcode_avail - n_bars))
    return np.fromiter(map(closest_bar, bars_per_barcode_match), dtype=np.int32)


def _match_bars(barcode1, barcode2):
    """Match bars between barcodes by length"""
    lengths1 = get_lengths(barcode1)
    lengths2 = get_lengths(barcode2)

    sorted_ids1 = np.argsort(lengths1)
    sorted_ids2 = np.argsort(lengths2)

    limit = min(sorted_ids1.size, sorted_ids2.size)

    mapping = np.full_like(sorted_ids1, dtype=np.int32, fill_value=-1)

    mapping[sorted_ids1[:limit]] = sorted_ids2[:limit]

    return mapping


def _find_in(paths, in_paths):

    paths = np.asarray(paths)
    in_paths = np.asarray(in_paths)

    if paths.size == in_paths.size and np.allclose(paths, in_paths):
        return np.arange(paths.size, dtype=np.int32)

    sorted_ids = np.argsort(in_paths)

    sorted_pos = np.searchsorted(in_paths[sorted_ids], paths)

    sorted_pos[sorted_pos < 0] = 0
    sorted_pos[sorted_pos >= len(in_paths)] = len(in_paths) - 1

    return sorted_ids[sorted_pos]


def _assign_er_to_sections(syn_barcodes, bio_barcodes, barcode_matching):

    section_ids_to_er_data = {}

    for syn_barcode_id, (syn_barcode, syn_barcode_data) in enumerate(syn_barcodes):

        # get the biological bar and its respective data that is matched
        # to the current synthetic bar
        bio_barcode, bio_barcode_data = bio_barcodes[barcode_matching[syn_barcode_id]]

        for syn_bar_id, bio_bar_id in enumerate(_match_bars(syn_barcode, bio_barcode)):

            if bio_bar_id == -1:
                continue

            bio_bar_data = bio_barcode_data[bio_bar_id]
            syn_bar_data = syn_barcode_data[syn_bar_id]

            bio_pdists, bio_er_data = bio_bar_data["ER"]
            syn_pdists = syn_bar_data["path_distances_from_terminations"]

            bio_ids = _find_in(syn_pdists, bio_pdists)

            # the section ids of the morphology that we need to assign er data
            section_ids = syn_bar_data["morphio_sid"]

            assert len(section_ids) == len(bio_ids)
            for syn_id, bio_id in enumerate(bio_ids):

                sid = section_ids[syn_id]
                er = bio_er_data[bio_id]

                if er is not None:
                    section_ids_to_er_data[sid] = er

    return section_ids_to_er_data
