# SPDX-License-Identifier: Apache-2.0

# pylint: disable-all

import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from bisect import bisect
from functools import reduce
from itertools import islice
from operator import mul

import numpy


class DataSetI(metaclass=ABCMeta):
    """
    Interface for datasets to be eligible as sources of CachedDataSet and ComposedCachedDataset.
    Notice the compatibility with numpy arrays, and eventually h5py.
    """

    @abstractmethod
    def __getitem__(self, group_name):
        pass

    def __setitem__(self, key, value):
        raise TypeError("Can't set attribute on such container")

    @abstractproperty
    def dtype(self):
        pass

    @abstractmethod
    def to_nparray(self):
        pass

    @abstractproperty
    def shape(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class CachingError(Exception):
    """
    An exception thrown when the caching system can't cope with the data requests.
    """

    pass


# ==================================================================================================
class CachedDataset(DataSetI):
    """
    A CachedDataset implements lazy loading of a very large dataset typically stored on disk, and
    optimizes reads by caching a relatively large block when needed, a kind of read-ahead buffer.
    """

    # ----------------------------------------------------------------------------------------------
    _MB = 1024 * 1024
    _MAX_DS_SIZE_MB = 1024

    # -----
    def __init__(self, ds, col_index=None, cache_size=4 * _MB, **kw):
        """
        Creates a new Cached Dataset from an existing data source implementing DataSetI
        (e.g. RawCStructReader, Numpy, ...)

        :param ds: The base dataset, anything implementing DataSetI
        :param col_index: (optional) Extract/cache a single column (field). Only available for
            non-CachedDataset
        :param cache_size: (optional) The cache size. Only available to non CachedDataset.
            Default: 4MB
        """
        if isinstance(ds, CachedDataset):
            self.__dict__.update(ds.__dict__)  # Get same properties as source
            self._slice_start = kw.get("_slice_start", ds._slice_start)
            self._slice_stop = kw.get("_slice_stop", ds._slice_stop)
            if col_index:
                if self._col is not None:
                    raise CachingError(
                        "CachedDataset can be column. "
                        "Convert to Numpy if you need advanced indexing."
                    )
                self._col = col_index

        else:
            self._ds = ds
            self._item_size = ds.dtype.itemsize
            if len(ds.shape) > 1:
                self._item_size *= reduce(mul, ds.shape[1:])
            self._base_cached_lines = cache_size // self._item_size
            self._ds_len = len(ds)
            self._cache = None
            self._cache_len = 0
            self._cache_offset = 0
            # Slice (virtual region)  - default all
            self._slice_start = 0  # Absolute offset of the selection
            self._slice_stop = self._ds_len  # Absolute end of the selection
            self._col = col_index

    # -----
    def __getitem__(self, item):
        """
        Returns an element or a view on the current DataSet using bracket [] syntax.

        :param item: The index, slice or column name \n
        |  If [integer] the request is immediately evaluated and a record returned.
        |  If [start:stop] a lazy Dataset) is returned
        |  If ["column"] a lazy Dataset containing only the selected column.
        Only once and single column is supported \n
        :return: a Numpy record, Numpy array, or CachedDataSet
        """
        if isinstance(item, slice):
            slice_len = self._slice_stop - self._slice_start
            actual_indices = item.indices(slice_len)  # Enable negative indexing
            # n_items = actual_indices[1] - actual_indices[0]
            new_start = self._slice_start + actual_indices[0]
            new_stop = self._slice_start + actual_indices[1]
            if new_start > self._ds_len or new_stop > self._ds_len:
                raise IndexError("Range %d-%d invalid" % actual_indices)
            # Be lazy, dont evaluate, return a new object with the right slicing
            return CachedDataset(self, _slice_start=new_start, _slice_stop=new_stop)

        elif isinstance(item, str):
            return CachedDataset(self, col_index=item)

        else:
            # A single item is cached and evaluated
            actual_index = self._slice_start + item
            self._check_load_buffer(actual_index)
            logging.debug("Asking number")
            item = self._cache[actual_index - self._cache_offset]
            if self._col is None:
                return item
            else:
                return item[self._col]

    # -----
    def __iter__(self):
        """An iterator on the entire dataset"""
        # Iterate the whole virtual region, eventually spanning over several buffers
        cur_slice_pos = self._slice_start
        while self._slice_stop > cur_slice_pos:
            self._check_load_buffer(cur_slice_pos)
            buffer_start = cur_slice_pos - self._cache_offset
            buffer_stop = min(self._cache_len, self._slice_stop - self._cache_offset)
            if self._col is None:
                for elem in islice(self._cache, buffer_start, buffer_stop):
                    yield elem
            else:
                for elem in islice(self._cache, buffer_start, buffer_stop):
                    yield elem[self._col]
            cur_slice_pos += buffer_stop - buffer_start

    # ----
    def to_nparray(self):
        """
        Materializes the current CachedDataset into a numpy array.\n
        :return: A numpy ndarray with all the data selected in the current dataset view.\n
        :raises CachingError: If the requested region is larger than CachedDataset._MAX_DS_SIZE_MB
        """
        requested_size = len(self) * self.dtype.itemsize // self._MB
        if requested_size > self._MAX_DS_SIZE_MB:
            raise CachingError(
                "The requested dataset exceeds {} MB maximum size (would require {} MB).\n"
                "  - Please select a smaller region or use iterators".format(
                    self._MAX_DS_SIZE_MB, requested_size
                )
            )

        # For little requests just use __iter__
        if len(self) < 100:
            full_array = numpy.empty(len(self), dtype=self.dtype)
            for i, rec in enumerate(self):
                full_array[i] = rec
        else:
            full_array = numpy.empty(0, dtype=self.dtype)
            for block in self._get_cache_blocks():
                full_array = numpy.concatenate((full_array, block))
        return full_array

    # -----
    def __len__(self):
        """
        The number of records of the current DataSet view. \n
        :Note: This is a virtual size, since records might not be in memory.
        """
        return self._slice_stop - self._slice_start

    @property
    def dtype(self):
        """:property: The same dtype as defined in the data source"""
        if self._col:
            return self._ds.dtype[self._col]
        else:
            return self._ds.dtype

    @property
    def shape(self):
        """:property: The shape of the dataset, basically tuple(length)"""
        return (len(self),)  # Tuple

    # ----
    def __repr__(self):
        return "<CachedDataset [{}:{}]>".format(self._slice_start, self._slice_stop)

    # ------------------------------------------------------------------------------ private section
    def _check_load_buffer(self, start_offset):
        """Verifies if a given virtual buffer position is available in memory,
        loading it if necessary.
        """
        end_offset = self._cache_offset + self._cache_len
        if start_offset < self._cache_offset or start_offset >= end_offset:
            # Align to groups of 32 elems - mitigate reverse access
            start_offset = (start_offset // 32) << 5
            end_offset = min(start_offset + self._base_cached_lines, self._ds_len)

            logging.debug("Loading buffer with range %d-%d", start_offset, end_offset)
            self._cache = self._ds[start_offset:end_offset]
            self._cache_offset = start_offset
            self._cache_len = end_offset - start_offset

    def _is_totally_cached(self):
        cache_end_offset = self._cache_offset + self._cache_len
        return self._slice_start >= self._cache_offset and self._slice_stop <= cache_end_offset

    def _get_cache_blocks(self):
        """Gets entire blocks from the source, bypassing cache, for fast conversion to numpy"""
        for start in range(self._slice_start, self._slice_stop, self._base_cached_lines):
            stop = min(start + self._base_cached_lines, self._slice_stop)
            logging.debug("Loading buffer with range %d-%d", start, stop)
            cache_b = self._ds[start:stop]
            if self._col is not None:
                cache_b = cache_b[self._col]
            yield cache_b


# ==================================================================================================
class ComposedCachedDataset(DataSetI):
    """
    A CachedDataset which handles multiple dataset sources and behaves as a contiguous container.
    """

    # ----------------------------------------------------------------------------------------------
    _MB = 1024 * 1024
    _MAX_DS_SIZE_MB = 1024

    def __init__(self, sub_containers):
        """
        :param sub_containers: A list of dataset sources, e.g. H5Py or RawCStructReader
        """
        self._container_count = 0
        self._containers = []
        self._offsets = []
        for c in sub_containers:
            self.add_container(c)
        # If subselection made, keep indexes just for info
        self._view_indexes = None

    def add_container(self, container):
        """
        Append an additional data source to the existing dataset.\n
        :param container: The additional data source to be incorporated, making the virtual dataset
        to increase length.
        """
        container = CachedDataset(container)
        cur_container_index = self._container_count
        self._containers.append(container)
        offset = len(container)
        if cur_container_index > 0:
            offset += self._offsets[cur_container_index - 1]
        self._offsets.append(offset)
        self._container_count += 1

    def __len__(self):
        """
        :returns: The size of the virtual container, in number of records
        """
        return sum(len(x) for x in self._containers)

    def __getitem__(self, item):
        """
        Returns an element or a view on the current DataSet using bracket [] syntax.
        Refer to **CachedDataset.__getitem__()** for a full explanation. \n
        :param item: An int, slice or column name.\n
        :return: a Numpy record, Numpy array, or CachedDataSet
        """
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            if step != 1:
                raise RuntimeError("PyTouchReader doesn't support slicing step != 1")
            c1_i, c1_offset = self._to_real_index(start)
            c2_i, c2_offset = self._to_real_index(stop)
            c1 = self._containers[c1_i]

            # If the data is in the same source container we return directly a slice of that
            # CachedContainer
            # Otherwise we return a new composed container with the subcontainers properly sliced
            if c1_i == c2_i:
                return c1[c1_offset:c2_offset]
            else:
                sub_ctns = [c1[c1_offset:]]
                for c_i in range(c1_i + 1, c2_i - 1):
                    sub_ctns.append(self._containers[c_i])
                sub_ctns.append(self._containers[c2_i][:c2_offset])
                return ComposedCachedDataset(sub_ctns)
        # Column name selection
        elif isinstance(item, str):
            return ComposedCachedDataset([ds[item] for ds in self._containers])
        # Single entry
        else:
            c_i, c_offset = self._to_real_index(item)
            return self._containers[c_i][c_offset]

    def __iter__(self):
        """An iterator on the entire dataset, all files considered"""
        for cnt in self._containers:
            for elem in cnt:
                yield elem

    def to_nparray(self):
        """
        loads all the records for the region selected by this dataset into a Numpy array. \n
        :returns: A Numpy ndarray
        """
        requested_size = len(self) * self.dtype.itemsize // self._MB
        if requested_size > self._MAX_DS_SIZE_MB:
            raise CachingError(
                "The requested dataset exceeds {} MB maximum size (would require {} MB).\n"
                "  - Please select a smaller region or use iterators".format(
                    self._MAX_DS_SIZE_MB, requested_size
                )
            )
        return numpy.concatenate([c.to_nparray() for c in self._containers])

    @property
    def shape(self):
        """:property: The virtual array shape, basically tuple(length)"""
        return (len(self),)  # Tuple

    @property
    def dtype(self):
        """:property: The records dtype, according to the source."""
        if not self._containers:
            raise CachingError("ComposedDataset without sources")
        return self._containers[0].dtype

    def __repr__(self):
        if self._view_indexes:
            return "<ComposedCachedDataset [{}]>".format(":".join(self._view_indexes))
        else:
            return "<ComposedCachedDataset (Length: {})>".format(len(self))

    # ------------------------------------------------------------------------- private
    def _to_real_index(self, i):
        container_i = bisect(self._offsets, i)
        if container_i == 0:
            offset = i
        else:
            offset = i - self._offsets[container_i - 1]
        return container_i, offset
