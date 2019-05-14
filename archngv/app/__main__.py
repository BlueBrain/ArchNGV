"""
Collection of ArchNGV tools.
"""

import logging
import click

from archngv.app import (
    cell_placement,
    microdomains,
    gliovascular_connectivity,
    neuroglial_connectivity,
    synaptic_connectivity,
    synthesis,
    assign_emodels,
    endfeet_area,
    convert,
)
from archngv.app.logger import LOGGER
from archngv.version import VERSION


def _setup_logging():
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.WARNING
    )
    LOGGER.setLevel(logging.INFO)


def main():
    """ Collection of tools for NGV building """
    _setup_logging()
    app = click.Group('ngv', {
        'cell-placement': cell_placement.cmd,
        'microdomains': microdomains.cmd,
        'gliovascular-connectivity': gliovascular_connectivity.cmd,
        'neuroglial-connectivity': neuroglial_connectivity.cmd,
        'synaptic-connectivity': synaptic_connectivity.cmd,
        'synthesis': synthesis.cmd,
        'assign-emodels': assign_emodels.cmd,
        'endfeet-area': endfeet_area.cmd,
        'convert': convert.app,
    })
    app = click.version_option(VERSION)(app)
    app()


if __name__ == '__main__':
    main()
