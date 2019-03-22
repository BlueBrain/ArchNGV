import numpy as np
from scipy import sparse, spatial
from .types import h5_point_map as pmap
from .types import h5_group_map as smap

from morphmath import apply_rotation_to_points


class Cell(object):
    """
    A Neuron object is a container for Trees and a Soma.
    The groups and points encode the 3D structure of the Neuron.
    """
    def __init__(self, name, points, groups):
        """General Cell Morphology Object

        Parameters:
            neuron: Obj neuron where groups and points are stored
            initial_direction: 3D vector or random
            initial_point:  the root of the tree
            radius: assuming that the radius is constant for now.
            tree_type: an integer indicating the type of the tree (choose from 2, 3, 4, 5)
        """
        self.name = name
        self._points = points
        self._groups = groups

    def types(self):
        raise NotImplementedError

    @property
    def n_sections(self):
        return len(self._groups)

    @property
    def n_points(self):
        return len(self.point_data)

    @property
    def point_data(self):
        return self._points

    @property
    def node_radii(self):
        return self._points[:, pmap['r']]

    @property
    def node_points(self):
        return self._points[:, pmap['xyz']]

    @property
    def section_types(self):
        return self._groups[:, smap['TYP']]

    @property
    def section_offsets(self):
        """ Index pointer to the points array for the start
        of all sections
        """
        return self._groups[:, smap['SO']]

    @property
    def section_parents(self):
        """ Parent section for each section
        """
        return self._groups[:, smap['PID']]

    @property
    def section_terminations(self):
        """ Returns all the terminations of all section from structure
        by accessing the next entry's offset or the end of the array
        """
        eidx = np.empty(self.n_sections, dtype=np.intp)

        # each section has as end the start of the next
        eidx[:-1] = self.section_offsets[1:]

        # except for the last one which ends at the total
        # points of the morphology
        eidx[-1] = self.n_points

        return eidx

    @property
    def soma_n_points(self):
        """ Number of points in the soma contour """
        return self.section_offsets[1] if self.n_sections > 1 else self.n_points

    @property
    def soma_points(self):
        """ The points of the soma """
        return self.node_points[:self.soma_n_points]

    @property
    def soma_radii(self):
        """ The radii of the points making up the soma """
        return np.linalg.norm(self.soma_points - self.soma_center, axis=1)

    @property
    def soma_center(self):
        """ Soma center calculated by averaging the distance
        to the center of the soma
        """
        return np.mean(self.soma_points, axis=0)

    @property
    def bounding_box(self):
        points = self.node_points
        return points.min(axis=0), points.max(axis=0)

    @property
    def node_adjacency_matrix(self):
        raise NotImplementedError

    def section_adjacency_matrix(self, include_soma=False):
        """ Returns the adjacency matrix of the sections of the cell

         the soma is not included as a section
        """
        n_sections = self.n_sections

        start_section = 0 if include_soma else 1

        # section ids apart from soma
        cidx = np.arange(1, n_sections, dtype=np.uintp)

        if cidx.size == 0:
            return None

        # their parents
        pidx = self.section_parents[cidx]

        # connectivity booleans
        data = np.ones(cidx.size, dtype=np.int)

        # sparse matrix creation
        A = sparse.csr_matrix((data, (pidx, cidx)), shape=(n_sections, n_sections), dtype=data.dtype)[1:, 1:]
        return AdjacencyMatrix(A, cidx)

    def sections_extrema(self, section_type=None):
        """ Returns the starting and ending points for each section. Without the soma included"""

        # section ids apart from the soma
        assert section_type is None or section_type in list(self.types.values()), list(self.types.values)

        mask = self.section_types == section_type if section_type is not None else Ellipsis

        return self.section_offsets[mask], self.section_terminations[mask]

    def sections(self, section_type=None):

        starts, ends = self.sections_extrema(section_type)

        return (self.point_data[start: end] for start, end in zip(starts, ends))

    def translate(self, point):
        self._points[:, pmap['xyz']] += point

    def rotate(self, rot_matrix):
        points = self._points[:, pmap['xyz']]
        self._points[:, pmap['xyz']] = apply_rotation_to_points(points, rot_matrix)

    def reset_soma_position(self):
        soma_center = self.soma_center
        self._points[:, pmap['xyz']] -= soma_center

    def section_closer_to_point(self, point):
        """ Returns the section which is closer to the given point
        """
        A = self.section_adjacency_matrix

        if A is None:
            return self.soma_points[-1], 0., 0

        # section terminations
        leaf_secs_idx = A.terminations

        # last point of section terminations
        end_point_idx = sections_ending_point_idx(leaf_secs_idx)

        # coordiantes fo the points
        points = self.node_points[end_point_idx]

        # pairwise distances
        distances = spatial.distance.cdist(points, point[np.newaxis])

        # index of min distance point to the given point
        index = np.argmin(distances)

        return points[index], self.node_radii[index], leaf_secs_idx[index]
