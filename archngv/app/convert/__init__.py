"""
Back and forth format conversion
"""

import click

from archngv.app.convert import (
    cell_data,
    neuroglial_connectivity
)


group = click.Group('convert', {
    'cell-data': cell_data.cmd,
    'neuroglial-connectivity': neuroglial_connectivity.cmd,
}, help=__doc__)
