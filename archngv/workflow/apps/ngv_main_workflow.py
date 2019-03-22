#!/usr/bin/env python

import logging
import argparse
from collections import namedtuple

from archngv.workflow.actions import building as _bld
from archngv.workflow.detail.apps_utils import execute_steps

L = logging.getLogger(__name__)

_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']


def _log_level_string_to_int(log_level_string):
    if not log_level_string in _LOG_LEVEL_STRINGS:
        message = 'invalid choice: {0} (choose from {1})'.format(log_level_string, _LOG_LEVEL_STRINGS)
        raise argparse.ArgumentTypeError(message)

    log_level_int = getattr(logging, log_level_string, logging.INFO)
    # check the logging log_level_choices have not changed from our expected values
    assert isinstance(log_level_int, int)

    return log_level_int


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Ngv config json file')
    parser.add_argument('-s', '--seed', dest='seed', type=int, help='Seed for random generators')
    parser.add_argument('--log-level',
                        default='INFO',
                        dest='log_level',
                        type=_log_level_string_to_int,
                        nargs='?',
                        help='Set the logging output level. {0}'.format(_LOG_LEVEL_STRINGS))

    names = \
    [
        'parallel',
        'all',
        'cell_placement',
        'microdomains',
        'gliovascular_connectivity',
        'neuroglial_connectivity',
        'endfeet_synthesis',
        'synthesis',
        'area_reconstruction'
    ]

    for name in names:
        parser.add_argument('--{}'.format(name), dest=name, action='store_true', default=False)

    return(parser.parse_args())


def selected_steps_iterator(args):

    BStep = namedtuple('BuildingStep', 'name is_enabled func')

    steps = [
    BStep('Cell Placement'             , args.cell_placement           , _bld.cell_placement)           ,
    BStep('Microdomain Generation'     , args.microdomains             , _bld.microdomain_tesselation)  ,
    BStep('Gliovascular Connectivity'  , args.gliovascular_connectivity, _bld.gliovascular_connectivity),
    BStep('Neuroglial Connectivity'    , args.neuroglial_connectivity  , _bld.neuroglial_connectivity)  ,
    BStep('Morphology Synthesis'       , args.synthesis                , _bld.morphology_synthesis)     ,
    BStep('Endfeet only synthesis'     , args.endfeet_synthesis        , _bld.astrocyte_endfeet_morphology_synthesis),
    BStep('Endfoot Area Reconstruction', args.area_reconstruction      , _bld.astrocyte_endfeet_area_reconstruction)
    ]

    return steps


if __name__ == '__main__':

    args = get_arguments()

    logging.basicConfig(level=args.log_level)

    steps_it = selected_steps_iterator(args)
    execute_steps(steps_it, args, 'Generation')
