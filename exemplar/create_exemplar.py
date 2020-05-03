import os
import stat
import shutil
import argparse
from pathlib import Path

from collections import namedtuple


def paths(target_directory):

    archngv_path = Path(__file__).resolve().parent.parent

    source_snakefile = archngv_path / 'snakemake/Snakefile'
    source_bioname_dir = archngv_path / 'exemplar/template/bioname'
    source_run_script = archngv_path / 'exemplar/template/run.sh'

    target_snakefile = target_directory / 'Snakefile'
    target_bioname_dir = target_directory / 'bioname'
    target_run_script = target_directory / 'run.sh'

    Paths = namedtuple('Paths', ['snakefile', 'bioname', 'run'])

    return (
        Paths(source_snakefile, source_bioname_dir, source_run_script),
        Paths(target_snakefile, target_bioname_dir, target_run_script)
    )

def parse_arguments():
    """ Argument parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('target',  type=Path)
    parser.add_argument('--create-links', action='store_true', default=False)
    return parser.parse_args()


def copy_and_overwrite(from_path, to_path):

    if from_path.is_dir():
        if to_path.exists():
            shutil.rmtree(to_path)
        shutil.copytree(from_path, to_path)
    else:
        shutil.copyfile(from_path, to_path)


if __name__ == '__main__':

    args = parse_arguments()
    target_directory = args.target
    create_links = args.create_links

    if not target_directory.exists():
        os.mkdir(target_directory)

    sources, targets = paths(target_directory)

    if create_links:

        os.symlink(sources.snakefile, targets.snakefile)
        os.symlink(sources.run, targets.run)
        os.symlink(sources.bioname, targets.bioname)

    else:

        copy_and_overwrite(sources.snakefile, targets.snakefile)
        copy_and_overwrite(sources.run, targets.run)
        copy_and_overwrite(sources.bioname, targets.bioname)

        # make target run script executable
        st = os.stat(targets.run)
        os.chmod(targets.run, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
