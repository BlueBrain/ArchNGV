# pylint: disable-all
from __future__ import absolute_import, division, print_function

import glob
import logging
import os.path
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from . import utils
from .caching import CachedDataset, ComposedCachedDataset, DataSetI

# Main and only public API class
__all__ = ["TouchInfo", "RawCStructReader"]


# =============================================================================================
class _TouchInfoI(metaclass=ABCMeta):
    """
    Interface offered by both TouchInfo and TouchInfoBase object
    """

    # ---------------------------------------------------------------------------------------------
    @abstractproperty
    def header(self):
        """
        :Property: the header record.

        Fields: (architectureIdentifier: float, numberOfNeurons: int, gitVersion: str)

        >>> tr.header
        (1.001, 10000L, '4.1.0')

        >>> tr.header.numberOfNeurons
        1000
        """
        pass

    @abstractproperty
    def neuron_count(self):
        """
        :Property: The number of neurons in the circuit.
        """
        pass

    @abstractproperty
    def neuron_stats(self):
        """
        :Property: a lazy :py:class:`~caching.CachedDataset` of all the neuron touch statistics.

        Each record has fields: (neuronID: int, touchesCount: int, binaryOffset: long)

        >>> tr.neuron_stats.dtype
        dtype([('neuronID', '>i4'), ('touchesCount', '>u4'), ('binaryOffset', '>i8')], align=True)

        >>> tr.neuron_stats[8000]
        (674588, 10279, 14403548200)

        >>> tr.neuron_stats[8000]["touchesCount"]
        10279

        |  When selecting ranges a :py:class:`~caching.CachedDataset` object is returned
        |  CachedDatasets support only a subset of Numpy operations. If required convert to a full \
        array using :py:meth:`~caching.CachedDataset.to_nparray`\n
        >>> tr.neuron_stats[8000:9000]
        <CachedDataset [8000:9000]>

        >>> tr.neuron_stats[100000:200000]
        <ComposedCachedDataset (Length: 100000)>   # Same but selection spans over multiple files
        """
        pass

    @abstractproperty
    def touch_count(self):
        """
        :Property: The total number of touches in this circuit
        """
        pass

    @abstractproperty
    def touches(self):
        """
        :Property: A virtual array (:py:class:`~caching.CachedDataset` - see example in \
        :py:attr:`~neuron_stats`) which will lazily load required raw touches when \
        slicing/indexing/iterating.

        Each record has fields: (pre_ids: [int]*3, post_ids: [int]*3, branch_order: \
        int, distances: [float]*3)\n
        |  NOTE: CachedDatasets are lazy and to be evaluated one must iterate over or explicitly \
        call to_nparray()
        |  Please select a small region (slice notation) before converting to a list or numpy array.
        """
        pass

    @abstractmethod
    def get_neuron_stats(self, gids):
        """
        Retrieves neuron_stats (number and offset of touches) given the gid(s).
        Gids that dont exist won't be in the dict, no exceptions are thrown.

        :param gids: A single or a list of gids
        :returns: A dictionary of {gid->neuron_info_record[fileIndex, touchesCount, binaryOffset]}

        >>> tr.get_neuron_stats(10)
        {10: (4, 59215, 260364760)}

        >>> tr.get_neuron_stats([10, 11, 12, 13])
        {10: (4, 59215, 260364760),
         11: (4, 54776, 262733360),
         12: (15, 43634, 341320320),
         13: (15, 44840, 343065680)}

        **Note**: Providing a single neuronID or list of neurons to look for incur similar \
        workload, since the entire dataset
        has to be read through eventually from disk. Therefore it is strongly advised to provide a \
        list of neuronIDs whenever
        there's a known set of interest. To loop through the *entire collection* it is preferable \
        to iterate over the
        :py:attr:`~neuron_stats` property.

        **Note2**: The first number is the fileIndex, which is not necessarily the same number in \
        the touch file extension.
        Therefore, with multiple touch files it is recommended to only consider the touch count.

        To access the touches for a given gid please consider using :py:meth:`~get_touches` instead
        """
        pass

    @abstractmethod
    def get_touches(self, pre_gid):
        """
        Retrieves the lazy CachedDataset of touches for a certain pre-synaptical neuron given its \
        ID.

        The result CachedDataset is then evaluated if converted to a numpy array (beware of memory \
        limits) or iterated over
        (minimal memory usage). The same way, to iterate over the *entire* collection of touches \
        the user might consider
        iterating over the :py:attr:`~TouchInfo.touches` property directly.

        :param pre_gid: The pre_gid of the touches to be collected
        :returns: a lazy :py:class:`~caching.CachedDataset` of touches for a given gid
        :raises KeyError: if the Gid doesn't exist

        For reference only, get_touches() is almost equivalent to:\n
        >>> # Please never write such code. There are no guarantees in the file ordering
        >>> file_i, count, offset = tr.get_neuron_stats(40)
        >>> TouchInfo("touches.{}".format(file_i)).touches[offset:offset+count]
        """
        pass


# ===========================================================================================================
class TouchInfo(_TouchInfoI):
    """A reader for Binary Touches, providing a easy interface to touch.x and tochData.x records"""

    # -----------------------------------------------------------------------------------------------------------
    NEURON_INFO_DTYPE = np.dtype(
        [("fileIndex", np.int32), ("touchesCount", np.uint32), ("binaryOffset", np.longlong)]
    )

    # ---
    def __init__(self, touch_file_or_dir):
        """
        Creates the main TouchInfo object. If a directory path is given it will consider all touch \
        files located inside. \n
        :param touchfile_or_dir: the path for a binary touch file or for a directory containing \
        several of them.

        >>> tr = pytouchreader.TouchInfo("/home/leite/scratch")
        >>> touches
        <PyTouchReader::TouchInfo:
          Path: /home/leite/scratch
          Number of source files: 16
          Number of neurons: 1984000>
        """
        self._source_path = touch_file_or_dir
        files = glob.glob(os.path.join(touch_file_or_dir, "touches.*"))
        files.sort(key=utils.natural_keys)
        if not files:
            raise ValueError("Path {} doesn't contain binary touch files".format(touch_file_or_dir))
        self._touch_infos = [BaseTouchInfo(f) for f in files]

    # If a single touch file is given to the constructor
    # we build and return a BaseTouchInfo instead
    def __new__(cls, touch_file_or_dir):
        if os.path.isfile(touch_file_or_dir):
            return BaseTouchInfo(touch_file_or_dir)
        else:
            return object.__new__(TouchInfo)

    @property
    def header(self):
        glob_header = self._touch_infos[0].header.copy()
        glob_header.numberOfNeurons = sum(ti.header.numberOfNeurons for ti in self._touch_infos)
        return glob_header

    @property
    def neuron_count(self):
        return self.header.numberOfNeurons

    @property
    def neuron_stats(self):
        return ComposedCachedDataset([info.neuron_stats for info in self._touch_infos])

    @property
    def touch_count(self):
        return sum(info.touch_count for info in self._touch_infos)

    @property
    def touches(self):
        return ComposedCachedDataset([info.touches for info in self._touch_infos])

    # ---
    def get_neuron_stats(self, gids):
        if not hasattr(gids, "__iter__") and not hasattr(gids, "__next__"):
            gids = [gids]
        gids = set(gids)
        result = {}
        for i, sub_touch_infos in enumerate(self._touch_infos):
            # Test if we already have everything
            if set(result) == gids:
                return result

            found_records = sub_touch_infos._get_neuron_stats_raw(gids)
            for record in found_records:
                gid = record["neuronID"]
                info = record.astype(self.NEURON_INFO_DTYPE)
                info["fileIndex"] = i
                result[gid] = info
        return result

    # ---
    def get_touches(self, pre_gid):
        stats = self.get_neuron_stats(pre_gid)
        if not stats:
            raise KeyError("pre_gid {} not found".format(pre_gid))
        stats = stats[pre_gid]
        index = stats["fileIndex"]
        start_offset = stats["binaryOffset"] // self._touch_infos[index].touchsize
        end_offset = start_offset + stats["touchesCount"]
        return self._touch_infos[index].touches[start_offset:end_offset]

    # ---
    def __repr__(self):
        return (
            "<PyTouchReader::TouchInfo:\n"
            "  Path: {}\n"
            "  Number of source files: {}\n"
            "  Number of neurons: {}>\n".format(
                self._source_path, len(self._touch_infos), self.neuron_count
            )
        )

    # Ugly fix docs (req py<3.5)
    for _name, _item in locals().copy().items():
        if _name.startswith("_"):
            continue
        if _item.__doc__ is None:
            try:
                _item.__doc__ = getattr(_TouchInfoI, _name).__doc__
            except Exception:
                pass


class Version(object):
    """
    Very basic representation of a version string
    """

    def __init__(self, s):
        try:
            self._elements = [int(n) for n in s.split(".")]
        except ValueError:
            self._elements = []
        while len(self._elements) > 0 and self._elements[-1] == 0:
            self._elements.pop(-1)

    def __ge__(self, other):
        for s, o in zip(self._elements, other._elements):
            if s > o:
                return True
            elif s < o:
                return False
        return len(self._elements) >= len(other._elements)

    def __str__(self):
        return ".".join(str(i) for i in self._elements)


# ==================================================================================================
class BaseTouchInfo(_TouchInfoI):
    """
    Reader of Binary Neuron-Touches info
    """

    # ----------------------------------------------------------------------------------------------
    _header_dtype = np.dtype(
        [
            ("architectureIdentifier", np.double),
            ("numberOfNeurons", np.longlong),
            ("version", "S16"),
        ],
        align=True,
    )

    _neuron_touches_dtype = np.dtype(
        [("neuronID", np.int32), ("touchesCount", np.uint32), ("binaryOffset", np.longlong)],
        align=True,
    )

    _touches_definitions = [
        (
            Version("5.4"),
            np.dtype(
                [
                    ("pre_ids", np.int32, 3),
                    ("post_ids", np.int32, 3),
                    ("branch_order", np.int32),
                    ("distances", np.float32, 3),
                    ("pre_section_fraction", np.float32),
                    ("post_section_fraction", np.float32),
                    ("pre_position", np.float32, 3),
                    ("post_position", np.float32, 3),
                    ("spine_length", np.float32),
                    ("branch_type", np.uint8),
                    ("pre_position_center", np.float32, 3),
                    ("post_position_surface", np.float32, 3),
                ],
                align=True,
            ),
        ),
        (
            Version("4.99"),
            np.dtype(
                [
                    ("pre_ids", np.int32, 3),
                    ("post_ids", np.int32, 3),
                    ("branch_order", np.int32),
                    ("distances", np.float32, 3),
                    ("pre_section_fraction", np.float32),
                    ("post_section_fraction", np.float32),
                    ("pre_position", np.float32, 3),
                    ("post_position", np.float32, 3),
                    ("spine_length", np.float32),
                    ("branch_type", np.uint8),
                ],
                align=True,
            ),
        ),
        (
            Version("0"),
            np.dtype(
                [
                    ("pre_ids", np.int32, 3),
                    ("post_ids", np.int32, 3),
                    ("branch_order", np.int32),
                    ("distances", np.float32, 3),
                ],
                align=True,
            ),
        ),
    ]

    _SIZEOF_HEADER = _header_dtype.itemsize
    _SIZEOF_NEURON_STATS = _neuron_touches_dtype.itemsize

    # ---
    def __init__(self, neuron_touches_file):
        # type: (str) -> None
        logging.debug("Loading " + neuron_touches_file)
        self._neuron_file = neuron_touches_file
        _last_dot = neuron_touches_file.rfind(".")
        self.touches_file = (
            neuron_touches_file[:_last_dot] + "Data" + neuron_touches_file[_last_dot:]
        )
        self._byte_swap = None
        self._file_handler = open(self._neuron_file, "rb")
        # Init header
        self._header = self._read_header(self._file_handler)
        self._version = self._get_version(self._header)
        self._touches_dtype = self._find_dtype(self._version)

    # ---
    def __del__(self):
        self._file_handler.close()

    @staticmethod
    def _get_version(header):
        """Extract version information from the header"""
        raw_version = header.version
        try:
            zero = raw_version.index(b"\x00")
            return Version(raw_version[:zero].decode())
        except ValueError:
            return Version(raw_version.decode())

    @classmethod
    def _find_dtype(cls, version):
        """Determine the version dependent touch datatype"""
        for v, dt in cls._touches_definitions:
            if version >= v:
                return dt
        msg = "Unknown touch version {0}".format(version)
        raise NotImplementedError(msg)

    @property
    def touchsize(self):
        """The byte count of the touch structure"""
        return self._touches_dtype.itemsize

    # ---
    def _read_header(self, fh):
        """Loads the initial header record"""
        try:
            header = np.rec.fromfile(fh, dtype=BaseTouchInfo._header_dtype, aligned=True, shape=1)[
                0
            ]
        except Exception as e:
            logging.fatal("Could not read header record from touches")
            raise e

        self._byte_swap = not np.equal(header.architectureIdentifier, np.double(1.001))
        if self._byte_swap:
            header.numberOfNeurons = header.numberOfNeurons.byteswap()
        # cache
        self._header = header
        return header

    @property
    def header(self):
        return self._header

    @property
    def neuron_count(self):
        return self._header.numberOfNeurons

    @property
    def neuron_stats(self):
        touchinfo_dataset = RawCStructReader(
            self._neuron_file, self._neuron_touches_dtype, self._byte_swap, self._SIZEOF_HEADER
        )
        return CachedDataset(touchinfo_dataset)

    @property
    def touch_count(self):
        return int(self.neuron_stats["touchesCount"].to_nparray().sum())

    # ---
    def _get_neuron_stats_raw(self, gids):
        gids = set(gids)
        return [stat for stat in self.neuron_stats if stat["neuronID"] in gids]

    # ---
    def get_neuron_stats(self, gids):
        """
        Get a dict of the stats records based on the gids
        :param gids: A single or a list of gids
        :return: A dictionary of {gid->_neuron_touches[neuronID, touchesCount, binaryOffset]}
        """
        if isinstance(gids, int):
            gids = [gids]
        gids = set(gids)
        result = {}
        for stat in self.neuron_stats:
            gid = int(stat["neuronID"])
            if stat["neuronID"] in gids:
                result[gid] = stat
        return result

    # ---
    @property
    def touches(self):
        touch_dataset = RawCStructReader(self.touches_file, self._touches_dtype, self._byte_swap)
        return CachedDataset(touch_dataset)

    # ---
    def get_touches(self, pre_gid):
        stats = self._get_neuron_stats_raw([pre_gid])
        if not stats:
            raise KeyError("pre_gid {} not found".format(pre_gid))
        stats = stats[0]
        start_offset = stats["binaryOffset"] // self.touchsize
        end_offset = start_offset + stats["touchesCount"]
        return self.touches[start_offset:end_offset]

    # ---
    def __repr__(self):
        return (
            "<PyTouchReader::TouchInfo:\n"
            "  Version: {}\n"
            "  DataType: {}\n"
            "  Path: {}\n"
            "  Number of neurons: {}\n>".format(
                self._version, self._touches_dtype, self._neuron_file, self.neuron_count
            )
        )


# ===============================================================================================
class RawCStructReader(DataSetI):
    """
    |  Generic C binary struct reader, providing a :py:class:`~caching.DataSetI` (array) interface.
    |  This class implements a raw reader, without caching or lazy evaluation.
    |  It is typically better used as a data source for a :py:class:`~caching.CachedDataset`
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, raw_file, record_dtype, swap_endianness=False, byte_offset=0, nrecords=None):
        """Initializes a RawStructReader \n
        :param raw_file: The path for the raw file
        :param record_dtype: The dtype of the elements to be read
        :param swap_endianness: (optional) If the records endianness must be swapped. Default: False
        :param byte_offset: (optional) Number of bytes to the beginning of the data block.
        :param nrecords: (optional) Number of records available. Defaults to calculate according to\
        file size
        """
        self.path = raw_file
        self._record_dtype = record_dtype if not swap_endianness else record_dtype.newbyteorder()
        self._record_size = record_dtype.itemsize
        self._byte_offset = byte_offset
        if nrecords is None:
            datablock_size = os.path.getsize(raw_file) - byte_offset
            self._length = datablock_size // self._record_size
        else:
            self._length = nrecords
        self._fh = open(raw_file, "rb")

    def __getitem__(self, index):
        """
        Fetches records from the buffer file into a numpy array using bracket [] syntax.\n
        :param index: The positions to be fetched (single int or slice notation)
        :return: np.ndarray with the records
        """
        if isinstance(index, slice):
            true_idx = index.indices(self._length)
            offset = true_idx[0]
            count = true_idx[1] - true_idx[0]
        elif isinstance(index, str):
            raise IndexError(
                "Index can't be a string. Column Selection not available in RawCStructReader"
            )
        else:
            offset = index
            count = 1
        self._fh.seek(self._byte_offset + offset * self._record_size)
        return np.fromfile(self._fh, dtype=self._record_dtype, count=count)

    def __setitem__(self, key, value):
        raise TypeError("Read-only structure")

    @property
    def dtype(self):
        """:property: The dtype."""
        return self._record_dtype

    @property
    def shape(self):
        """:property: the shape, basically tuple(length)."""
        return (self._length,)  # tuple

    def __len__(self):
        """:returns: The number of records available"""
        return self._length

    def to_nparray(self):
        """
        :returns: the whole selected buffer (according to constructor params) as a numpy array.\n
        **Beware** the source is totally loaded into memory.
        For handling large files please consider using CachedDataSet on top of this object
        """
        return self[:]  # Force __getitem__ which returns a nparray
