""" Exporters for the cell placement stage of the NGV pipeline
"""


def hdf5_exporter(filepath, astrocyte_ids, somata_positions, somata_radii):
    """Exporter using the hdf5 library.

    Args:
        filepath: string
            Absolute path to data file.
        astrocyte_ids: array[int, (N,)]
        somata_positions: array[float, (N, 3)]
        somata_radii: array[float, (N,)]
    """
    import h5py

    with h5py.File(filepath, "w") as fd:
        fd.create_dataset("cell_ids", data=astrocyte_ids)
        fd.create_dataset("positions", data=somata_positions)
        fd.create_dataset("radii", data=somata_radii)


def basalt_exporter(filepath, astrocyte_ids, somata_positions, somata_radii):
    """Exporter using the basalt framework.

    Args:
        filepath:
            Absolute path to data file.
        astrocyte_ids: array[int, (N,)]
        somata_positions: array[float, (N, 3)]
        somata_radii: array[float, (N,)]
    """
    from basalt.ngv import NGVGraph, Point  # pylint: disable=import-error

    ngv_graph = NGVGraph(path=filepath)
    astrocytes = ngv_graph.astrocytes  # pylint: disable=no-member

    for i, astrocyte_id in enumerate(astrocyte_ids):

        astrocyte = astrocytes[astrocyte_id]
        payload = astrocyte.data

        payload.soma_center = Point(
            somata_positions[i, 0], somata_positions[i, 1], somata_positions[i, 2]
        )
        payload.soma_radius = somata_radii[i]

        ngv_graph.vertices.add((astrocyte.type.value, astrocyte.id), payload.serialize())

    ngv_graph.commit()


def export_cell_placement_data(
    filepath, astrocyte_ids, somata_positions, somata_radii, method="hdf5"
):
    """
    Args:
        filepath: string
            Full path and name of output file
        astrocyte_ids: array[int, (N,)]
            The ids of the cells.
        somata_positions: array[float, (N, 3)]
        somata_radii: array[float, (N,)]
        method: string
            Export method. 'hdf5' or 'basalt'
    """
    writers = {"hdf5": hdf5_exporter, "basalt": basalt_exporter}
    try:
        writer_func = writers[method]
    except KeyError as e:
        msg = f"Export method {method} is not valid. Choose from {writers.keys()}"
        raise KeyError(msg) from e
    else:
        writer_func(filepath, astrocyte_ids, somata_positions, somata_radii)
