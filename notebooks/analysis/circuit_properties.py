import os
from pathlib import Path
import logging
from functools import partial
import multiprocessing
import json
import h5py

from collections import namedtuple
import numpy as np
import pandas as pd
from archngv import NGVConfig, NGVCircuit
from bluepysnap import Circuit
from cached_property import cached_property
from morphio import Morphology

from bluepysnap.bbp import Cell
from circuit_interface import CircuitInterface


L = logging.getLogger(__name__)


class CircuitProperties(CircuitInterface):

    def astrocyte_segment_index(self, filepath):
        """ Loaf or create a segment index of the circuit astrocytes """
        if Path(filepath).exists():
            from segment_index import SegmentIndex
            L.info('Astrocytes segment index already exists and will be opened from file.')
            return SegmentIndex(filepath)
        from segment_index import create_cell_segment_index
        filepaths = self.astrocyte_filepaths
        # astrocytes are already in space!
        positions = np.zeros((len(filepaths), 3), dtype=np.float32)
        return create_cell_segment_index(
            filepaths, positions, filepath, is_astrocyte=True)

    def neuronal_segment_index(self, filepath):
        """ Loaf or create a segment index of the circuit neurons """
        if Path(filepath).exists():
            from segment_index import SegmentIndex
            L.info('Neurons segment index already exists and will be opened from file.')
            return SegmentIndex(filepath)

        from segment_index import create_cell_segment_index
        return create_cell_segment_index(
            self.neuronal_filepaths, self.neuronal_positions, filepath, is_astrocyte=False)

    def vasculature_segment_index(self, filepath):
        """ Loaf or create a segment index of the circuit vasculature """
        if Path(filepath).exists():
            from segment_index import SegmentIndex
            L.info('Vasculature segment index already exists and will be opened from file.')
            return SegmentIndex(filepath)

        from segment_index import create_vasculature_segment_index
        return create_vasculature_segment_index(self.circuit.data.vasculature, filepath)

    def segment_features_per_layer(self,
        neurons_segment_index, astrocytes_segment_index, vasculature_segment_index):
        from analytics import segment_statistics_per_layer
        return segment_statistics_per_layer(
            self.bbox,
            neurons_segment_index,
            astrocytes_segment_index,
            vasculature_segment_index
        )

    def segment_features_per_microdomain(self, index, samples_per_layer=10):
        from analytics import segment_statistics_per_microdomain
        return segment_statistics_per_microdomain(
            self.overlapping_microdomains,
            self.astrocyte_positions,
            index,
            samples_per_layer
        )


    @cached_property
    def primary_processes_per_astrocyte(self):
        n_astrocytes = self.n_astrocytes

        counts = np.zeros(n_astrocytes, dtype=np.int)

        for i in range(n_astrocytes):
            try:
                root_sections = self.morphology(i).root_sections
            except:
                continue
            counts[i] = len(root_sections)

        return counts

    @cached_property
    def astrocyte_process_stats(self):
        n_astrocytes = self.n_astrocytes
        type_counts_per_astro = np.zeros((n_astrocytes, 2), dtype=np.int)

        map_types = {2: 0, 3: 1}

        for i in range(n_astrocytes):
            try:
                root_sections = self.morphology(i).root_sections
            except:
                continue

            s_types = [int(s.type) for s in root_sections]
            unique_types, counts = np.unique(s_types, return_counts=True)

            assert len(unique_types) <= 2, unique_types

            for t, count in zip(unique_types, counts):
                type_counts_per_astro[i, map_types[t]] = count

        return type_counts_per_astro

    @cached_property
    def microdomain_centroids(self):
        norm_tess = self.microdomains
        over_tess = self.overlapping_microdomains

        norm_centroids = np.asarray((d.centroid for d in norm_tess))
        over_centroids = np.asarray((d.centroid for d in over_tess))

        return norm_centroids, over_centroids

    @cached_property
    def astrocyte_somata_volumes(self):
        return (4. / 3.) * np.pi * self.astrocyte_radii ** 3

    @cached_property
    def number_of_endfeet_per_astrocyte(self):
        it = map(self.endfeet_ids_for_astrocyte, range(self.n_astrocytes))
        return np.fromiter((len(ids) for ids in it), dtype=np.int)

    @cached_property
    def endfeet_path_lengths(self):
        analysis_file = os.path.join(self.config.parent_directory, 'build/analysis/analysis.json')
        with open(analysis_file, 'r') as fd:
            data = json.load(fd)
        return np.asarray(data['path_lengths'])
