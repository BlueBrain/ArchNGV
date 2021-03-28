import os
import logging
from pathlib import Path
from numpy import testing as npt

import pytest


L = logging.getLogger(__name__)


BUILD_DIR = Path(__file__).parent.resolve() / 'build'
EXPECTED_DIR = Path(__file__).parent.resolve() / 'expected'

SONATA_DIR = './sonata'
MORPHOLOGIES_DIR = './morphologies'
MICRODOMAINS_DIR = './microdomains'


def _get_h5_files(directory):

    filenames = os.listdir(directory)
    filenames = filter(lambda s: s.endswith('.h5'), filenames)

    return sorted(filenames)


def _filenames_verify_cardinality(actual_directory, expected_directory):
    """Return the expected filenames and check if the produced filenames
    are identical in number and names.
    """
    actual_filenames = _get_h5_files(actual_directory)
    desired_filenames = _get_h5_files(expected_directory)

    npt.assert_equal(actual_filenames, desired_filenames, err_msg=(
        f'Differing output filenames:\n'
    ))

    return desired_filenames


def test_morphologies():
    """ Compare synthesized hdf5 morphologies
    """
    from morph_tool.morphio_diff import diff

    filenames = _filenames_verify_cardinality(
        BUILD_DIR / MORPHOLOGIES_DIR,
        EXPECTED_DIR / MORPHOLOGIES_DIR
    )

    for filename in filenames:

        diff_result = diff(
            BUILD_DIR / MORPHOLOGIES_DIR / filename,
            EXPECTED_DIR / MORPHOLOGIES_DIR / filename
        )

        assert not diff_result, diff_result.info


def _h5_compare(actual_filepath, expected_filepath):

    import subprocess

    completed_process = subprocess.run(
        ['h5diff', '-v', '-c', '--delta=5e-07', actual_filepath, expected_filepath]
    )

    assert completed_process.returncode == 0


def _h5_compare_all(actual_dir, expected_dir):
    for filename in _filenames_verify_cardinality(actual_dir, expected_dir):
        _h5_compare(actual_dir / filename, expected_dir / filename)


def test_sonata_files():

    _h5_compare_all(
        BUILD_DIR / SONATA_DIR / 'nodes',
        EXPECTED_DIR / SONATA_DIR / 'nodes'
    )

    _h5_compare_all(
        BUILD_DIR / SONATA_DIR / 'edges',
        EXPECTED_DIR / SONATA_DIR / 'edges'
    )


def test_microdomain_files():

    _h5_compare_all(
        BUILD_DIR / MICRODOMAINS_DIR,
        EXPECTED_DIR / MICRODOMAINS_DIR
    )


def test_root_files():
    """Files at the root level of build-expected"""
    _h5_compare_all(BUILD_DIR, EXPECTED_DIR)
