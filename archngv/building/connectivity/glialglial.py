""" Generation of connectivity of glial with their
neighbors via the formation of gap junctions
"""

from typing import Tuple
import logging
import numpy as np
import pandas as pd


L = logging.getLogger(__name__)


def _edges_from_touchreader(touches_directory: str) -> np.ndarray:
    """
    Use pytouchreader to load gap junctions and extract the edges
    between pre_ids and post_ids
    """
    from pytouchreader import TouchInfo  # pylint: disable=import-error

    touches = TouchInfo(touches_directory).touches

    return np.column_stack((
        touches['pre_ids'].to_nparray()[:, 0],  # [cell_id, section_id, segment_id]
        touches['post_ids'].to_nparray()[:, 0]
    ))


def _symmetric_connections_and_ids(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Loads the touchdetector touches and creates symmetric
    unique connections between astrocytes

    Returns:
        symmetric_edges:
            Undirected connection edges between astrocytes
        symmetric_ids:
            The ids corresponding to the edges. Symmetric edges share the same
            id because they represent the same connection.
    """
    if len(edges) == 0:
        L.warning('No touches have been found.')
        return np.array([]), np.array([])

    # sort by column to swamp any symmetric edges
    # [1, 0], [0, 1] -> [0, 1], [0, 1]
    edges.sort(axis=1)

    # remove row duplicates
    edges = np.unique(edges, axis=0)

    n_connections = len(edges)
    edge_ids = np.arange(n_connections, dtype=np.int)

    # touches are stored one way in the touchdetector
    # thus we concatenate all the symmetric edges
    symmetric_edges = np.vstack((edges, edges[:, [1, 0]]))

    # and take care to have the same id for symmetric edges
    symmetric_ids = np.hstack((edge_ids, edge_ids))

    # mirror entries to the lower triangular matrix
    return symmetric_edges, symmetric_ids


def _glialglial_dataframe(symmetric_edges: np.ndarray, connection_ids: np.ndarray):
    """
    Create a glial glial connectivity dataframe
    """
    if len(symmetric_edges) > 0:
        astrocyte_source_ids, astrocyte_target_ids = symmetric_edges.T
    else:
        astrocyte_source_ids = astrocyte_target_ids = np.array([])

    df = pd.DataFrame({'astrocyte_source_id': astrocyte_source_ids,
                       'astrocyte_target_id': astrocyte_target_ids,
                       'connection_id': connection_ids})

    df.sort_values(['astrocyte_target_id', 'astrocyte_source_id', 'connection_id'], inplace=True)
    return df


def generate_glialglial(touches_directory: str):
    """ Create glial glial connectivity dataframe """
    # TODO : the touches are actually something we should be able to access.
    # this file will probably change from unique connections to touches.
    edges = _edges_from_touchreader(touches_directory)
    edges, edges_ids = _symmetric_connections_and_ids(edges)
    return _glialglial_dataframe(edges, edges_ids)
