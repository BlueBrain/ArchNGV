import time
import logging
import numpy

from archngv import NGVConfig

L = logging.getLogger('__name__')


def process_random_seed(ngv_config):
    try:
        seed = ngv_config._config['seed']
        numpy.random.seed(seed=seed)
        L.info('seed provided: {}'.format(seed))
    except KeyError:
        L.info('No seed provided.')


def _execute_step(step, ngv_config, map_func):

    process_random_seed(ngv_config)

    L.info('\n{} started.'.format(step.name))

    ts = time.time()

    step.func(ngv_config, map_func)

    te = time.time()

    L.info('{} completed. Elapsed time: {}'.format(step.name, ts - te))
    L.info('\n')


def execute_steps(steps, args, type_of_execution):

    L.info('\nArguments: {}\n'.format(args))

    ngv_config = NGVConfig.from_file(args.config)

    L.info('{} of Experiment {} started.'.format(type_of_execution, ngv_config.experiment_name))

    msg = 'Selected steps: ' + ', '.join([step.name for step in steps if step.is_enabled ^ args.all])
    L.info(msg)

    for step in steps:

        # xor allows us to also selectively disable a step in the list
        # e.g. --run_all --run_cell_placement will run everythin but cell placement

        if step.is_enabled ^ args.all:
            _execute_step(step, ngv_config, args.parallel)

    L.info('{} of Experiment {} was successful'.format(type_of_execution, ngv_config.experiment_name))
