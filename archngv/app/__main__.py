r"""
Collection of tools for NGV building

{esc}
   _____                .__      _______    ____________   ____
  /  _  \_______   ____ |  |__   \      \  /  _____/\   \ /   /
 /  /_\  \_  __ \_/ ___\|  |  \  /   |   \/   \  ___ \   Y   /
/    |    \  | \/\  \___|   Y  \/    |    \    \_\  \ \     /
\____|__  /__|    \___  >___|  /\____|__  /\______  /  \___/
        \/            \/     \/         \/        \/
"""
import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import pkg_resources

import click
from archngv.version import VERSION

from archngv.app.commands import (
    convert,
    cell_placement,
    assign_emodels,
    finalize_astrocytes,
    microdomains,
    gliovascular,
    neuroglial,
    glialglial_connectivity,
    synthesis,
    endfeet_area,
    ngv_config
)

from archngv.app.logger import setup_logging


@click.group('ngv', help=__doc__.format(esc='\b'))
@click.version_option(version=VERSION)
@click.option("-v", "--verbose", count=True, default=0, help="-v for INFO, -vv for DEBUG")
def app(verbose):
    # pylint: disable=missing-docstring
    setup_logging(
        {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
        }[min(verbose, 2)]
    )


app.add_command(name='convert', cmd=convert.group)  # see commands.convert.__init__.py
app.add_command(name='cell-placement', cmd=cell_placement.cmd)
app.add_command(name='assign-emodels', cmd=assign_emodels.cmd)
app.add_command(name='finalize-astrocytes', cmd=finalize_astrocytes.cmd)
app.add_command(name='microdomains', cmd=microdomains.cmd)
app.add_command(name='gliovascular', cmd=gliovascular.group)
app.add_command(name='neuroglial', cmd=neuroglial.group)
app.add_command(name='glialglial-connectivity', cmd=glialglial_connectivity.cmd)
app.add_command(name='synthesis', cmd=synthesis.cmd)
app.add_command(name='endfeet-area', cmd=endfeet_area.cmd)
app.add_command(name='config-file', cmd=ngv_config.cmd)


@app.command(name='snakefile-path')
def snakefile_path():
    """Outputs a path to the default Snakefile."""
    click.echo(pkg_resources.resource_filename(__name__, 'snakemake/Snakefile'))


def _index(args, *opts):
    """Finds index position of `opts` in `args`"""
    indices = [i for i, arg in enumerate(args) if arg in opts]
    assert len(indices) < 2, f"{opts} options can't be used together, use only one of them"
    if len(indices) == 0:
        return None
    return indices[0]


def _build_args(args, bioname, modules, timestamp):
    if _index(args, '--printshellcmds', '-p') is None:
        args = ['--printshellcmds'] + args
    if _index(args, '--cores', '--jobs', '-j') is None:
        args = ['--jobs', '8'] + args
    # force the timestamp to the same value in different executions of snakemake
    args = args + ['--config', f'bioname={bioname}', f'timestamp={timestamp}']
    if modules:
        raise NotImplementedError('This feature is not available yet')
        # serialize the list of strings with json to be backward compatible with Snakemake:
        # snakemake >= 5.28.0 loads config using yaml.BaseLoader,
        # snakemake < 5.28.0 loads config using eval.
        # args += [f'modules={json.dumps(modules)}']
    return args


def _run_snakemake_process(cmd, errorcode=1):
    """Run the main snakemake process."""
    from archngv.app.logger import LOGGER
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOGGER.error("Snakemake process failed")
        return errorcode
    return 0


def _run_summary_process(cmd, filepath: Path, errorcode=2):
    """Save the summary to file."""
    from archngv.app.logger import LOGGER
    cmd = cmd + ['--detailed-summary']
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w') as fd:
        result = subprocess.run(cmd, stdout=fd, check=False)
    if result.returncode != 0:
        LOGGER.error("Summary process failed")
        return errorcode
    return 0


def _run_report_process(cmd, filepath: Path, errorcode=4):
    """Save the report to file."""
    from archngv.app.logger import LOGGER
    cmd = cmd + ['--report', str(filepath)]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOGGER.error("Report process failed")
        return errorcode
    return 0


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option(
    '-u', '--cluster-config', required=True, type=click.Path(exists=True, dir_okay=False),
    help='Path to cluster config.',
)
@click.option(
    '--bioname', required=True, type=click.Path(exists=True, file_okay=False),
    help='Path to `bioname` folder of a circuit.',
)
@click.option(
    '-m', '--module', 'modules', multiple=True, required=False,
    help='''
Modules to be overwritten. Multiple configurations are allowed, and each one
should be given in the format:\n
    module_env:module_name/module_version[,module_name/module_version...][:module_path]\n
Examples:\n
    brainbuilder:archive/2020-08,brainbuilder/0.14.0\n
    touchdetector:archive/2020-05,touchdetector/5.4.0,hpe-mpi\n
    spykfunc:archive/2020-06,spykfunc/0.15.6:/gpfs/bbp.cscs.ch/ssd/apps/hpc/jenkins/modules/all
    '''
)
@click.option(
    '-s', '--snakefile', required=False, type=click.Path(exists=True, dir_okay=False),
    default=pkg_resources.resource_filename(__name__, 'snakemake/Snakefile'), show_default=True,
    help='Path to workflow definition in form of a snakefile.',
)
@click.option(
    '--with-summary', is_flag=True, help='Save a summary in `logs/<timestamp>/summary.tsv`.'
)
@click.option(
    '--with-report', is_flag=True, help='Save a report in `logs/<timestamp>/report.html`.'
)
@click.pass_context
def run(
    ctx,
    cluster_config: str,
    bioname: str,
    modules: list,
    snakefile: str,
    with_summary: bool,
    with_report: bool,
):
    """Run a circuit-build task.

    Any additional snakemake arguments or options can be passed at the end of this command's call.
    """
    from archngv.app.logger import LOGGER
    args = ctx.args
    if snakefile is None:
        snakefile = pkg_resources.resource_filename(__name__, 'snakemake/Snakefile')
    assert Path(snakefile).is_file(), f'Snakefile "{snakefile}" does not exist!'
    assert _index(args, '--config', '-C') is None, 'snakemake `--config` option is not allowed'

    timestamp = f"{datetime.now():%Y%m%dT%H%M%S}"
    args = _build_args(args, bioname, modules, timestamp)

    cmd = ['snakemake', *args, '--snakefile', snakefile, '--cluster-config', cluster_config]
    exit_code = _run_snakemake_process(cmd)

    if with_summary:

        # snakemake with the --summary/--detailed-summary option does not execute the workflow
        filepath = Path(f'logs/{timestamp}/summary.tsv')
        LOGGER.info("Creating report in %s", filepath)
        exit_code += _run_summary_process(cmd, filepath)

    if with_report:

        # snakemake with the --report option does not execute the workflow
        filepath = Path(f'logs/{timestamp}/report.html')
        LOGGER.info("Creating summary in %s", filepath)
        exit_code += _run_report_process(cmd, filepath)

    # cumulative exit code given by the union of the exit codes, only for internal use
    #   0: success
    #   1: snakemake process failed
    #   2: summary process failed
    #   4: report process failed
    sys.exit(exit_code)


if __name__ == '__main__':
    app()  # pylint: disable=no-value-for-parameter
