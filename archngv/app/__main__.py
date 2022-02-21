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
import logging
import os
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click
import pkg_resources

from archngv.app import ngv
from archngv.app.logger import setup_logging
from archngv.version import VERSION


@click.group("ngv", help=__doc__.format(esc="\b"))
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


app.add_command(name="cell-placement", cmd=ngv.cell_placement)
app.add_command(name="assign-emodels", cmd=ngv.assign_emodels)
app.add_command(name="finalize-astrocytes", cmd=ngv.finalize_astrocytes)
app.add_command(name="microdomains", cmd=ngv.build_microdomains)
app.add_command(name="gliovascular-connectivity", cmd=ngv.gliovascular_connectivity)
app.add_command(
    name="attach-endfeet-info-to-gliovascular-connectivity",
    cmd=ngv.attach_endfeet_info_to_gliovascular_connectivity,
)
app.add_command(name="neuroglial-connectivity", cmd=ngv.neuroglial_connectivity)
app.add_command(
    name="attach-morphology-info-to-neuroglial-connectivity",
    cmd=ngv.attach_morphology_info_to_neuroglial_connectivity,
)


app.add_command(name="synthesis", cmd=ngv.synthesis)
app.add_command(name="glialglial-connectivity", cmd=ngv.build_glialglial_connectivity)
app.add_command(name="endfeet-area", cmd=ngv.build_endfeet_surface_meshes)
app.add_command(name="config-file", cmd=ngv.ngv_config)


@app.command(name="create-exemplar")
@click.argument("project-dir", type=Path)
def create_exemplar(project_dir):
    """Create an exemplar circuit to build"""
    import shutil

    def copy_and_overwrite(from_path, to_path):
        if from_path.is_dir():
            if to_path.exists():
                shutil.rmtree(to_path)
            shutil.copytree(from_path, to_path)
        else:
            shutil.copyfile(from_path, to_path)

    if not project_dir.exists():
        os.mkdir(project_dir)

    exemplar_dir = Path(pkg_resources.resource_filename(__name__, "exemplar"))

    copy_and_overwrite(exemplar_dir / "bioname", project_dir / "bioname")
    copy_and_overwrite(exemplar_dir / "run.sh", project_dir / "run.sh")
    copy_and_overwrite(exemplar_dir / "launch.sbatch", project_dir / "launch.sbatch")

    # make run script executable
    st = os.stat(project_dir / "run.sh")
    os.chmod(project_dir / "run.sh", st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@app.command(name="snakefile-path")
def snakefile_path():
    """Outputs a path to the default Snakefile."""
    click.echo(pkg_resources.resource_filename(__name__, "snakemake/Snakefile"))


def _index(args, *opts):
    """Finds index position of `opts` in `args`"""
    indices = [i for i, arg in enumerate(args) if arg in opts]
    assert len(indices) < 2, f"{opts} options can't be used together, use only one of them"
    if len(indices) == 0:
        return None
    return indices[0]


def _build_args(args, bioname, timestamp):
    if _index(args, "--printshellcmds", "-p") is None:
        args = ["--printshellcmds"] + args
    if _index(args, "--cores", "--jobs", "-j") is None:
        args = ["--jobs", "8"] + args
    # force the timestamp to the same value in different executions of snakemake
    args = args + ["--config", f"bioname={bioname}", f"timestamp={timestamp}"]
    return args


def _run_snakemake_process(cmd, errorcode=1):
    """Run the main snakemake process."""
    from archngv.app.logger import LOGGER

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOGGER.error("Snakemake process failed")
        return errorcode
    return 0


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option(
    "-u",
    "--cluster-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to cluster config.",
)
@click.option(
    "--bioname",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to `bioname` folder of a circuit.",
)
@click.option(
    "-s",
    "--snakefile",
    required=False,
    type=click.Path(exists=True, dir_okay=False),
    default=pkg_resources.resource_filename(__name__, "snakemake/Snakefile"),
    show_default=True,
    help="Path to workflow definition in form of a snakefile.",
)
@click.pass_context
def run(
    ctx,
    cluster_config: str,
    bioname: str,
    snakefile: str,
):
    """Run a circuit-build task.

    Any additional snakemake arguments or options can be passed at the end of this command's call.
    """
    args = ctx.args
    if snakefile is None:
        snakefile = pkg_resources.resource_filename(__name__, "snakemake/Snakefile")
    assert Path(snakefile).is_file(), f'Snakefile "{snakefile}" does not exist!'
    assert _index(args, "--config", "-C") is None, "snakemake `--config` option is not allowed"

    timestamp = f"{datetime.now():%Y%m%dT%H%M%S}"
    args = _build_args(args, bioname, timestamp)

    cmd = [
        "snakemake",
        *args,
        "--snakefile",
        snakefile,
        "--cluster-config",
        cluster_config,
    ]
    exit_code = _run_snakemake_process(cmd)

    # cumulative exit code given by the union of the exit codes, only for internal use
    #   0: success
    #   1: snakemake process failed
    #   2: summary process failed
    #   4: report process failed
    sys.exit(exit_code)


if __name__ == "__main__":
    app()  # pylint: disable=no-value-for-parameter
