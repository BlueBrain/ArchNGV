#!/usr/bin/env python

import logging
import argparse
from collections import namedtuple

from archngv.workflow.actions import preprocessing as _pre
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
        'rewrite_vasculature_mesh'
    ]

    for name in names:
        parser.add_argument('--{}'.format(name), dest=name, action='store_true', default=False)

    return(parser.parse_args())


def selected_steps_iterator(args):

    # all available steps
    BStep = namedtuple('BuildingStep', 'name is_enabled func')

    steps = [
    BStep('Rewrite Vasculature Mesh' , args.rewrite_vasculature_mesh, _pre.rewrite_vasculature_mesh)
    ]

    return steps


if __name__ == '__main__':

    args = get_arguments()
    steps_it = selected_steps_iterator(args)
    execute_steps(steps_it, args, 'Preprocessing')
