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
    neuroglial,
    synthesis,
    assign_emodels,
    endfeet_area,
    convert,
)
from archngv.app.logger import setup_logging
from archngv.version import VERSION


@click.group('ngv', help=__doc__.format(esc='\b'))
@click.option("-v", "--verbose", count=True, help="-v for INFO, -vv for DEBUG")
@click.version_option(VERSION)
def app(verbose=0):
    # pylint: disable=missing-docstring
    level = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }[verbose]
    setup_logging(level)


app.add_command(name='cell-placement', cmd=cell_placement.cmd)
app.add_command(name='microdomains', cmd=microdomains.cmd)
app.add_command(name='gliovascular-connectivity', cmd=gliovascular_connectivity.cmd)
app.add_command(name='neuroglial', cmd=neuroglial.group)
app.add_command(name='synthesis', cmd=synthesis.cmd)
app.add_command(name='assign-emodels', cmd=assign_emodels.cmd)
app.add_command(name='endfeet-area', cmd=endfeet_area.cmd)
app.add_command(name='convert', cmd=convert.group)


if __name__ == '__main__':
    app()
