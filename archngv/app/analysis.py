""" Analysis of generated circuit """
import json
import click


class AstrocyteSubsetParallelWorker:
    "worker"
    def __init__(self, config, analysis_function):
        self.config = config
        self.analysis_function = analysis_function

    def __call__(self, astrocyte_ids):
        """ Given a set of ids, load the circuit and apply the
        analysis function on it."""
        from archngv import NGVCircuit
        ngv_circuit = NGVCircuit(self.config)
        return self.analysis_function(ngv_circuit, astrocyte_ids)


def _batch_ids(n_batches, total_elements):
    '''batch ids'''
    return (list(range(i, min(i + n_batches, total_elements)))
            for i in range(0, total_elements, n_batches))


@click.command(help=__doc__)
@click.option("--ngv-config", help="Path to the circuit config", required=True)
@click.option("--seed", help="Pseudo-random generator seed", type=int, default=0, show_default=True)
@click.option("-o", "--output", help="Path to output json", required=True)
def cmd(ngv_config, seed, output):
    """ Analysis of generated circuit """
    import multiprocessing
    import numpy as np
    from archngv import NGVConfig, NGVCircuit
    from archngv.extras.analysis.endfeet_morphometrics import endfeet_morphometrics

    config = NGVConfig.from_file(ngv_config)
    n_elements = len(NGVCircuit(config).data.astrocytes)

    n_cpus = multiprocessing.cpu_count()
    n_batches = n_elements // n_cpus

    analysis_functions = [endfeet_morphometrics]

    results = {}

    np.random.seed(seed)

    with multiprocessing.Pool(n_cpus) as pool:
        for analysis_function in analysis_functions:

            worker = AstrocyteSubsetParallelWorker(config, analysis_function)

            for worker_results in pool.imap(worker, _batch_ids(n_batches, n_elements)):
                for key, values in worker_results.items():
                    try:
                        results[key].extend(values)
                    except KeyError:
                        results[key] = values

    with open(output, 'w') as fd:
        json.dump(results, fd, indent=2)
