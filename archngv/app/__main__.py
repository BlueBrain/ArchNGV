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
    # pylint: disable=missing-docstring
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
        'convert': convert.group,
    }, help=__doc__.format(esc='\b'))
    app = click.version_option(VERSION)(app)
    app()


if __name__ == '__main__':
    main()
