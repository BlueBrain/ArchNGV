"""
App for finalizing the sonata glia cells by merging the different parts
"""
import click


@click.command(help=__doc__)
@click.option("--somata-file", help="Path to sonata somata file", required=True)
@click.option("--emodels-file", help="Path to sonata emodels file", required=True)
@click.option("-o", "--output", help="Path to output HDF5", required=True)
def cmd(somata_file, emodels_file, output):
    """Build the astrocytes population by merging the different astrocyte fields"""
    from voxcell import CellCollection
    from archngv.core.constants import Population
    somata = CellCollection.load(somata_file)
    emodels = CellCollection.load(emodels_file)

    somata.properties['model_template'] = emodels.properties['model_template']
    somata.population_name = Population.ASTROCYTES
    somata.save_sonata(output)
