""" Contains the vasculature class """

import h5py
import numpy as np

from .section import Section
from .types import pmap, emap, sconmap
from ..util.bounding_box import BoundingBox
from .graph.graphs import DirectedGraph
from .transformations import remap_edge_vertices


class Vasculature(object):
    """ Class container for vasculature datasets
    """

    @classmethod
    def load(cls, filename):
        """ Vasculature constructor from h5 file
        """
        with h5py.File(filename, 'r') as h5f:

            point_data = h5f['points'][:]

            edge_data = h5f['edges'][:]

            sections_structure = h5f['chains']['structure'][:]

            sections_connectivity = h5f['chains']['connectivity'][:]

        return cls(point_data, edge_data, sections_structure, sections_connectivity, annotations=None)

    def __init__(self, point_data,
                       edge_data,
                       sections_structure,
                       sections_connectivity,
                       annotations=None):

        self._edge_data = edge_data
        self._point_data = point_data
        self._section_structure = sections_structure
        self._section_connectivity = sections_connectivity

        if annotations is not None:

            self._annotations = annotations
        else:
            self._annotations = {}

    def __str__(self):
        return "points: {}\nedges:{}".format(self._point_data.shape[0], self._edge_data.shape[0])

    def __iter__(self):
        return iter(self.sections)

    @property
    def annotations(self):
        """ annotations property """
        return self._annotations

    @property
    def point_data(self):
        """ Get x, y, z, radius point data """
        return self._point_data

    @property
    def edge_data(self):
        """ Get edges, types data """
        return self._edge_data

    @property
    def section_structure(self):
        """ Get section structure """
        return self._section_structure

    @property
    def points(self):
        """ Get x, y, z, points """
        return self.point_data[:, pmap['xyz']]

    @points.setter
    def points(self, values):
        """ xyz """
        assert self.points.shape == values.shape
        self.point_data[:, pmap['xyz']] = values

    @property
    def n_points(self):
        """ Number of points"""
        return len(self.points)

    @property
    def radii(self):
        """ Get point node radii """
        return self.point_data[:, pmap['r']]

    @property
    def edges(self):
        """ Get connectivity edges """
        return self._edge_data[:, emap['edges']]

    @property
    def edge_types(self):
        """ Get types of edges """
        return self._edge_data[:, emap['type']]

    @property
    def section_connectivity(self):
        """ Get connectivity of sections """
        return self._section_connectivity[:, sconmap['edges']]

    @property
    def map_edges_to_sections(self):
        """ Returns section connectivity based on edges """
        connectivity = np.zeros(len(self.edges), dtype=np.intp)

        so = self.section_structure

        n_sections = len(so)

        for i in range(n_sections - 1):

            connectivity[so[i]: so[i + 1]] = i

        # use the last i that has bled from the loop
        connectivity[so[n_sections - 1]::] = i + 1

        return connectivity

    @property
    def sections(self):
        """ Get list of vasculature sections """
        edata = self.edge_data
        pdata = self.point_data

        # edge indices
        eidx = np.arange(len(edata), dtype=np.uintp)

        soff = self.section_structure

        n_offsets = len(soff)

        sec_objs = np.empty(n_offsets, dtype=np.object)

        for annotation, p_cloud in self.annotations.items():

            sorted_p = p_cloud.properties.sort('edge_index')

        for i in range(n_offsets - 1):

            i_edata = edata[soff[i]: soff[i + 1]]
            i_eidx = eidx[soff[i]: soff[i + 1]]

            # remaping is used because these edges
            # will be contained in each section
            # and they will point to iits subset of points
            r_edata = remap_edge_vertices(i_edata)

            i_pdata = pdata[np.unique(i_edata.ravel())]

            sec_annotation = {}

            for annotation, p_cloud in self.annotations.items():

                mask = sorted_p['edge_index'].isin(i_eidx)

                if mask.any():
                    sec_annotation[annotation] = sorted_p[mask]
                    sorted_p = sorted_p[~mask]

            sec_objs[i] = Section(i_pdata, r_edata, i_eidx, sec_annotation)

        last_edata = edata[soff[n_offsets - 1]: pdata.shape[0]]
        last_rdata = remap_edge_vertices(last_edata)
        last_pdata = pdata[np.unique(last_edata.ravel())]

        last_eidx = eidx[soff[n_offsets - 1]: pdata.shape[0]]

        sec_annotation = {}

        for annotation, p_cloud in self.annotations.items():

            mask = sorted_p['edge_index'].isin(i_eidx)
            if mask.any():
                sec_annotation[annotation] = sorted_p[mask]

        sec_objs[n_offsets - 1] = Section(last_pdata, last_rdata, last_eidx, sec_annotation)

        return sec_objs

    @property
    def point_graph(self):
        """ Returns the graph representation of the connectivity of the dataset """
        return DirectedGraph(self.edges, labels=None)

    @property
    def section_graph(self):
        """  Get the graph of the section connectivity """
        return DirectedGraph(self.section_connectivity, labels=self.sections)

    @property
    def segments(self):
        """
        A segment is a set of two 3D points connected with an edge
        Returns two 1D arrays with the begining and the end of a segment
        """
        edges = self.edges
        points = self.points
        return points[edges[:, 0]], \
               points[edges[:, 1]]

    @property
    def segments_radii(self):
        """ Returns the start and end radii of the segments """
        edges = self.edges
        radii = self.radii
        return radii[edges[:, 0]], \
               radii[edges[:, 1]]

    @property
    def segments_types(self):
        """ Get types of segments. Alias for edge types """
        return self.edge_types

    @property
    def bounding_box(self):
        """ Returns bb object """
        return BoundingBox.from_points(self.points)

    def spatial_index(self):
        """ Returns vasculature spatial index """
        from spatial_index import sphere_rtree
        return sphere_rtree(self.points, self.radii)

    def save(self, filename):
        """ Save vasculature to h5 file """
        if not filename.endswith('.h5'):

            filename += '.h5'

        with h5py.File(filename, 'w') as h5f:

            h5f.create_dataset('points', data=self.point_data)
            h5f.create_dataset('edges', data=self._edge_data)

            chain_group = h5f.create_group('chains')
            chain_group.create_dataset('structure',
                                       data=self.section_structure.astype(np.intp))
            chain_group.create_dataset('connectivity',
                                       data=self._section_connectivity.astype(np.intp))
