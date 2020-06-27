"""Vasculature dataset conversion to SONATA NodePopulation cli"""

import click
from archngv.app.utils import REQUIRED_PATH


@click.command(help=__doc__)
@click.argument('vasculature', type=REQUIRED_PATH)
@click.argument('output-file', type=str)
def cmd(vasculature, output_file):
    """Convert a vasculature section spec to a vasculatre NodePopulation edges one"""
    from archngv.core.datasets import Vasculature
    vasculature = Vasculature.load(vasculature)
    vasculature.save_sonata(output_file)
