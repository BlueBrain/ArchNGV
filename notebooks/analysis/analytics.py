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


