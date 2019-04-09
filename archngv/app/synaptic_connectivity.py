import click


@click.command()
@click.option("--circuit", help="Path to CircuitConfig", required=True)
@click.option("--output-connectivity", help="Path to output HDF5 (connectivity)", required=True)
@click.option("--output-data", help="Path to output HDF5 (data)", required=True)
def cmd(circuit, output_connectivity, output_data):
    from bluepy.v2 import Circuit

    from archngv.core.exporters import export_neuron_synapse_association
    from archngv.app.logger import LOGGER

    circuit = Circuit(circuit)

    LOGGER.info('Started exporting associations...')
    export_neuron_synapse_association(
        circuit,
        output_path_connectivity=output_connectivity,
        output_path_data=output_data,
    )
