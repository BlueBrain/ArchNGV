#!/usr/bin/env python

import logging
import argparse
from collections import namedtuple

from archngv.workflow.actions import postprocessing as _post
from archngv.workflow.detail.apps_utils import execute_steps

L = logging.getLogger(__name__)


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Ngv config json file')
    parser.add_argument('-s', '--seed', dest='seed', type=int, help='Seed for random generators')

    names = \
    [
        'parallel',
        'all',
        'associate_synapses_astrocytes'
    ]

    for name in names:
        parser.add_argument('--{}'.format(name), dest=name, action='store_true', default=False)

    return(parser.parse_args())


def selected_steps_iterator(args):

    # all available steps
    BStep = namedtuple('BuildingStep', 'name is_enabled func')

    steps = [
    BStep('Associate Synapses with astrocytes' , args.associate_synapses_astrocytes, _post.associate_synapses_with_morphology)
    ]

    return steps


if __name__ == '__main__':

    args = get_arguments()
    steps_it = selected_steps_iterator(args)
    execute_steps(steps_it, args, 'Postprocessing')
