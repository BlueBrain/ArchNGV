"""h5 readers are simple standalone sonata readers using libsonata that prevents in memory
loading of data and does not need circuit vision/config files."""
import numpy as np
from cached_property import cached_property
import libsonata

from archngv.exceptions import NGVError
from archngv.utils.generics import ensure_list


def _open_population(h5_filepath, pop_type):
    storage = {
        'nodes': libsonata.NodeStorage,
        'edges': libsonata.EdgeStorage
    }[pop_type](h5_filepath)

    populations = storage.population_names
    if len(populations) != 1:
        raise NGVError(
            "Only single-population edge collections are supported (found: %d)" % len(populations)
        )
    return storage.open_population(list(populations)[0])


class LibSonataReader:
    """Sonata Population context manager"""

    def __init__(self, filepath, population_type):
        self.filepath = filepath
        # pylint: disable=attribute-defined-outside-init
        self._impl = _open_population(filepath, pop_type=population_type)

    @property
    def name(self):
        """Return the name of the population"""
        return self._impl.name

    @cached_property
    def property_names(self):
        """Set of available properties inside the file."""
        return set(self._impl.attribute_names)

    def _selection(self, ids=None):
        """Returns a libsonata selection from a list or a single ids"""
        if ids is None:
            return self._impl.select_all()
        if isinstance(ids, libsonata.Selection):
            return ids
        if isinstance(ids, (np.integer, int)):
            ids = [ids]
        return libsonata.Selection(ids)

    def get_source_nodes(self, ids=None):
        """Returns source nodes"""
        return self._impl.source_nodes(self._selection(ids=ids)).astype(np.int64)

    def get_property(self, property_name, ids=None):
        """Returns a numpy array containing the values corresponding to property_name"""
        selection = self._selection(ids=ids)
        try:
            return self._impl.get_attribute(property_name, selection)
        except libsonata.SonataError as e:
            raise NGVError(f'Unknown property name {property_name}') from e

    def get_properties(self, property_names, ids=None):
        """Returns a numpy array containing the values corresponding to property_names"""
        ids = self._selection(ids)
        props = ensure_list(property_names)
        properties = [self.get_property(p, ids=ids) for p in props]
        if all(prop.dtype == properties[0].dtype for prop in properties):
            return np.column_stack([self.get_property(p, ids=ids) for p in props])
        raise NGVError("Can't stack properties with different types {}".format(
            [prop.dtype for prop in properties]))

    def __len__(self):
        return self._impl.size


class EdgesReader(LibSonataReader):
    """ Context manager for accessing SONATA Edges """
    def __init__(self, filepath):
        super().__init__(filepath, 'edges')

    def afferent_edges(self, node_ids):
        """Returns the edge ids heading to node_ids"""
        return self._impl.afferent_edges(node_ids).flatten()

    def efferent_edges(self, node_ids):
        """Returns the edge ids leaving the node_ids"""
        return self._impl.efferent_edges(node_ids).flatten()

    def efferent_nodes(self, node_ids, unique=True):
        """Returns efferent node IDs for given sources ``node_ids``.

        Args:
            node_ids (int/list): Source node ID.
            unique (bool): If ``True``, return only unique efferent node IDs.

        Returns:
            numpy.ndarray: Efferent node IDs.
        """
        selection = self._impl.efferent_edges(node_ids)
        result = self._impl.target_nodes(selection)
        if unique:
            result = np.unique(result)
        return result

    def afferent_nodes(self, node_ids, unique=True):
        """Returns afferent node IDs for given targets ``node_ids``.

        Args:
            node_ids (int/list): target node ID
            unique (bool): If ``True``, return only unique afferent node IDs.

        Returns:
            numpy.ndarray: Afferent node IDs.
        """
        selection = self._impl.afferent_edges(node_ids)
        result = self._impl.source_nodes(selection)
        if unique:
            result = np.unique(result)
        return result


class NodesReader(LibSonataReader):
    """ Context manager for accessing SONATA Nodes """
    def __init__(self, filepath):
        super().__init__(filepath, 'nodes')
