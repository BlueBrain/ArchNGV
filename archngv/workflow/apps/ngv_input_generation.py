#!/usr/bin/env python

import logging
import argparse
from collections import namedtuple

from archngv.workflow.actions import input_generation as _inp
from archngv.workflow.detail.apps_utils import execute_steps

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Ngv config json file')
    parser.add_argument('-s', '--seed', dest='seed', type=int, help='Seed for random generators')

    names = \
    [
        'parallel',
        'all',
        'neuronal_somata_geometries',
        'vasculature_spatial_index',
        'neuronal_somata_spatial_index',
        'neuronal_synapses_spatial_index',
        'neuron_synapse_connectivity'
    ]

    for name in names:
        parser.add_argument('--{}'.format(name), dest=name, action='store_true', default=False)

    return(parser.parse_args())


def selected_steps_iterator(args):

    # all available steps
    BStep = namedtuple('BuildingStep', 'name is_enabled func')

    steps = [
    BStep('Extract Neuronal Somata'       , args.neuronal_somata_geometries       , _inp.neuronal_somata_geometries),
    BStep('Vasculature Spatial Index'     , args.vasculature_spatial_index        , _inp.vasculature_spatial_index),
    BStep('Neu Somata Spatial Index'      , args.neuronal_somata_spatial_index    , _inp.neuronal_somata_spatial_index),
    BStep('Synapses Spatial Index'        , args.neuronal_synapses_spatial_index  , _inp.neuronal_synapses_spatial_index),
    BStep('Neuron - Synapse Connectivity' , args.neuron_synapse_connectivity      , _inp.neuron_synapse_connectivity)
    ]

    return steps


if __name__ == '__main__':

    args = get_arguments()
    steps_it = selected_steps_iterator(args)
    execute_steps(steps_it, args, 'Input Generation')
