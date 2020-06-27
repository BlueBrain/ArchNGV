"""
Back and forth format conversion
"""

import click

from archngv.app.convert import (
    cell_data,
    merge_sonata,
    ngv_config,
    sonata_config,
    sonata_vasculature
)


group = click.Group('convert', {
    'cell-data': cell_data.cmd,
    'ngv-config': ngv_config.cmd,
    'merge-sonata': merge_sonata.cmd,
    'sonata-config': sonata_config.cmd,
    'sonata-vasculature': sonata_vasculature.cmd
}, help=__doc__)
