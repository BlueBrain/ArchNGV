import logging
import numpy as np
import pandas as pd
from common import Layers


L = logging.getLogger(__name__)


class Entity:

    Neuron = 0
    Astrocyte = 1
    Vasculature = 2


ENTITY_NAMES = {
    Entity.Neuron: 'neuron',
    Entity.Astrocyte: 'astrocyte',
    Entity.Vasculature: 'vasculature'
}


class Stat:

    Length = 0
    Area = 1
    Volume = 2


STAT_NAMES = {
    Stat.Length: 'length',
    Stat.Area: 'area',
    Stat.Volume: 'volume'
}


def segment_statistics_per_layer(bbox, neurons_segment_index, astrocytes_segment_index, vasculature_segment_index):
 
    layers = Layers()
    thicknesses = layers.thicknesses()[::-1]
    labels = layers.labels[::-1]

    min_point, max_point = bbox.ranges
   
    xmin, ymin, zmin = min_point
    xmax, ymax, zmax = max_point

    depth = ymax

    indexes = {
        Entity.Neuron: neurons_segment_index,
        Entity.Astrocyte: astrocytes_segment_index,
        Entity.Vasculature: vasculature_segment_index
    }

    d = {'entity': [], 'layer': [], 'length': [], 'area': [], 'volume': []}

    for layer, thickness in enumerate(thicknesses):

        new_depth = depth - thickness
        L.info(f'Layer: {new_depth, depth}')

        for entity, index in indexes.items():
        
            L.info(f'Entity: {ENTITY_NAMES[entity]}')
            stats = index.intersection(xmin, new_depth, zmin, xmax, depth, zmax)

            for stat_index, stat_name in STAT_NAMES.items():
                d[stat_name].append(stats[stat_index])

            d['layer'].append(labels[layer])
            d['entity'].append(ENTITY_NAMES[entity])

        depth = new_depth

    return pd.DataFrame.from_dict(d)


def segment_statistics_per_microdomain(domains, domain_centers, segment_index, samples_per_layer):
    layers = Layers()
    bins = layers.bins
    labels = layers.labels

    ids = np.arange(len(domain_centers), dtype=np.int)

    d = {'astrocyte': [], 'layer': [], 'length': [], 'area': [], 'volume': []}

    for i in range(6):

        mask = (domain_centers[:, 1] > bins[i]) & (domain_centers[:, 1] < bins[i + 1])

        layer_ids = ids[mask]
        sample_ids = np.random.choice(layer_ids, size=samples_per_layer, replace=False)

        L.info(f'{len(layer_ids)} found in layer {labels[i]}. A sample of {len(sample_ids)} is chosen.')
        for index in sample_ids:
            L.info(f'Layer: {labels[i]}, Index: {index}')

            domain = domains[index]
            xmin, ymin, zmin, xmax, ymax, zmax = domain.bounding_box

            stats = segment_index.intersection_polygon(
                xmin, ymin, zmin, xmax, ymax, zmax, domain.face_points, domain.face_normals)

            for stat_index, stat_name in STAT_NAMES.items():
                d[stat_name].append(stats[stat_index])

            d['layer'].append(labels[i])
            d['astrocyte'].append(index)

    return pd.DataFrame.from_dict(d)

