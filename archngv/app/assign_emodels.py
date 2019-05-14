import click


@click.command()
@click.option("-i", "--input", help="Path to input SONATA Nodes HDF5", required=True)
@click.option("--hoc", help="HOC template file name", required=True)
@click.option("-o", "--output", help="Path to output SONATA Nodes HDF5", required=True)
def cmd(input, hoc, output):
    """Assign 'model_template' attribute"""
    from voxcell.sonata import NodePopulation

    nodes = NodePopulation.load(input)
    nodes.attributes['model_template'] = 'hoc:%s' % hoc
    nodes.save(output)