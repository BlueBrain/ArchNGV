
import os
import sys
import h5py
import logging
from cached_property import cached_property
import numpy as np

from archngv.core.data_structures.data_cells import CellData
from archngv.core.data_structures.data_synaptic import SynapticData
from archngv.core.data_structures.data_gliovascular import GliovascularData
from archngv.core.data_structures.data_microdomains import MicrodomainTesselation

from archngv.core.data_structures.connectivity_synaptic import SynapticConnectivity
from archngv.core.data_structures.connectivity_neuroglial import NeuroglialConnectivity
from archngv.core.data_structures.connectivity_gliovascular import GliovascularConnectivity

from archngv.core.exporters.export_gliovascular_connectivity import write_endfoot_perspective
from archngv.core.exporters.export_gliovascular_connectivity import write_astrocyte_perspective
from archngv.core.exporters.export_gliovascular_connectivity import write_vasculature_segment_perspective

logging.basicConfig(level=logging.DEBUG)

L = logging.getLogger(__name__)


class MockCellData(object):

    def __init__(self, filename, n_cells):
        self.filename = filename
        self.n_cells = n_cells

    @property
    def cell_names(self):
        return ['cell_{}'.format(i) for i in range(self.n_cells)]

    @property
    def cell_ids(self):
        return np.arange(self.n_cells, dtype=np.uintp)

    @property
    def positions(self):
        vals = np.arange(self.n_cells, dtype=np.float)
        return np.column_stack((vals, vals, vals))

    @property
    def radii(self):
        return np.arange(self.n_cells, dtype=np.float)

    def export(self):

        names = np.array(self.cell_names, dtype=h5py.special_dtype(vlen=str))

        with h5py.File(self.filename, 'w') as fd:

            fd.create_dataset('ids', data=self.cell_ids)

            fd.create_dataset('names', data=names)

            fd.create_dataset('positions', data=self.positions)
            fd.create_dataset('radii', data=self.radii)

    def test(self):

        with CellData(self.filename) as cell_data:

            assert cell_data.n_cells == self.n_cells
            assert np.allclose(cell_data.astrocyte_point_data, np.column_stack((self.positions, self.radii)))

            assert np.allclose(cell_data.astrocyte_positions, self.positions)
            assert np.allclose(cell_data.astrocyte_radii, self.radii)

            assert all([name == t_name for name, t_name in zip(cell_data.astrocyte_names, self.cell_names)])


###########################################################################################################3
###########################################################################################################3
###########################################################################################################3


class MockMicrodomainTesselation(object):

    def __init__(self, filename, n_cells):
        self.filename = filename
        self.n_cells = n_cells
        self.n_neighbors = 4

    def domain_points(self, index):
        vals = np.arange(10, dtype=np.float)
        return float(index) + np.column_stack((vals, vals, vals))

    def domain_triangles(self, index):
        return [(0, 1, 2),
                (2, 3, 4),
                (4, 5, 6),
                (6, 7, 8), 
                (9, 0, 1)]

    def domain_neighbors(self, index):
        return np.arange(self.n_neighbors, dtype=np.uintp)

    @property
    def microdomain_connectivity(self):
        return [(i, j) for i in range(self.n_cells) for j in range(self.n_neighbors) if i != j]

    def export(self):

        flat_connectivity = self.microdomain_connectivity

        with h5py.File(self.filename, 'w') as fd:

            data_group = fd.create_group('Data')

            data_points = []
            data_triangles = []
            data_neighbors = []

            offsets = np.zeros((n_cells + 1, 3), dtype=np.uintp)
            current_offsets = np.zeros(3, dtype=np.uintp)

            for index in range(n_cells):

                cell_points = self.domain_points(index)
                cell_triangles = self.domain_triangles(index)
                cell_neighbors = self.domain_neighbors(index)

                data_triangles.extend(cell_triangles)

                current_offsets[0] += len(cell_points)
                current_offsets[1] += len(cell_triangles)
                current_offsets[2] += len(cell_neighbors)

                offsets[index + 1] = current_offsets

                data_points.extend(cell_points)
                data_neighbors.extend(cell_neighbors)

            data_group.create_dataset('points', data=data_points, dtype=np.float)
            data_group.create_dataset('triangles', data=data_triangles, dtype=np.uintp)
            data_group.create_dataset('neighbors', data=data_neighbors, dtype=np.uintp)

            offsets_dset = fd.create_dataset('offsets', data=offsets, dtype=np.uintp)
            offsets_dset.attrs["column_names"] = \
            np.array(['points', 'triangles', 'neighbors'], dtype=h5py.special_dtype(vlen=str))

            conn_dset = fd.create_dataset('connectivity', data=flat_connectivity, dtype=np.uintp)
            conn_dset.attrs["column_names"] = \
            np.array(['i_domain_index', 'j_domain_index'], dtype=h5py.special_dtype(vlen=str))

    def test(self):

        with MicrodomainTesselation(self.filename) as mdom:

            assert mdom.n_microdomains == self.n_cells

            for astrocyte_index in range(self.n_cells):
                assert mdom.n_neighbors(astrocyte_index) == self.n_neighbors
                assert np.allclose(mdom.domain_neighbors(astrocyte_index),
                                   self.domain_neighbors(astrocyte_index))

                assert np.allclose(mdom.domain_points(astrocyte_index),
                                   self.domain_points(astrocyte_index))

                assert np.allclose(mdom.domain_triangles(astrocyte_index),
                                   self.domain_triangles(astrocyte_index))


###########################################################################################################
###########################################################################################################
###########################################################################################################


class MockGliovascularConnectivity(object):

        def __init__(self, filename, n_cells, n_endfeet):
            self.filename = filename
            self.n_cells = n_cells
            self.n_endfeet = n_endfeet
            self.n_vasculature_segments = n_endfeet

        @cached_property
        def endfeet_per_astrocyte(self):
            endfeet = np.arange(self.n_endfeet, dtype=np.uintp)
            return np.split(endfeet, self.n_cells)

        @property
        def vasculature_segments_per_astrocyte(self):
            return self.endfeet_per_astrocyte

        @cached_property
        def endfoot_to_astrocyte(self):

            res = np.zeros(self.n_endfeet, dtype=np.uintp)

            offset = 0
            for astrocyte_index, endfeet in enumerate(self.endfeet_per_astrocyte):

                n_endfeet = len(endfeet)
                res[offset: offset + n_endfeet] = astrocyte_index
                offset += n_endfeet

            return res

        @cached_property
        def endfoot_to_vasculature(self):
            return np.arange(self.n_endfeet, dtype=np.uintp)

        def export(self):

            astrocyte_to_endfeet = self.endfeet_per_astrocyte
            endfeet_to_astrocyte = self.endfoot_to_astrocyte
            endfeet_to_vasculature = self.endfoot_to_vasculature

            with h5py.File(self.filename, 'w') as fd:

                write_endfoot_perspective(fd, endfeet_to_astrocyte, endfeet_to_vasculature)
                write_astrocyte_perspective(fd, astrocyte_to_endfeet, endfeet_to_vasculature)
                write_vasculature_segment_perspective(fd, endfeet_to_astrocyte, endfeet_to_vasculature)

        def test(self):

            endfeet_per_astrocyte = self.endfeet_per_astrocyte
            va_segs_per_astrocyte = self.vasculature_segments_per_astrocyte

            with GliovascularConnectivity(self.filename) as conn:

                assert conn.n_astrocytes == self.n_cells
                assert conn.n_endfeet == self.n_endfeet

                for astrocyte_index in range(self.n_cells):

                    assert np.allclose(conn.astrocyte.to_endfoot(astrocyte_index),
                                       endfeet_per_astrocyte[astrocyte_index])

                    assert np.allclose(conn.astrocyte.to_vasculature_segment(astrocyte_index),
                                       va_segs_per_astrocyte[astrocyte_index])

                endfeet_to_astrocyte = self.endfoot_to_astrocyte
                endfeet_to_vasc = self.endfoot_to_vasculature

                for endfoot_index in range(self.n_endfeet):

                    assert conn.endfoot.to_astrocyte(endfoot_index) == endfeet_to_astrocyte[endfoot_index]
                    assert conn.endfoot.to_vasculature_segment(endfoot_index) == endfeet_to_vasc[endfoot_index]

                for seg_index in range(self.n_vasculature_segments):

                    assert conn.vasculature_segment.to_endfoot(seg_index) == seg_index
                    assert conn.vasculature_segment.to_astrocyte(seg_index) == endfeet_to_astrocyte[seg_index]

                assert(conn.vasculature_segment.to_endfoot(self.n_vasculature_segments) == None)
                assert(conn.vasculature_segment.to_astrocyte(self.n_vasculature_segments) == None)

###########################################################################################################
###########################################################################################################
###########################################################################################################


class MockGliovascularData(object):


    def __init__(self, filename, n_cells, n_endfeet):
        self.filename = filename
        self.n_cells = n_cells
        self.n_endfeet = n_endfeet

    def graph_coordinates(self):
        vals = np.arange(n_endfeet, dtype=np.float)
        return np.column_stack((vals, vals, vals))        

    def surface_coordinates(self):
        return self.graph_coordinates + 2.0

    def export(self):

        with h5py.File(self.filename, 'w') as fd:
            fd.create_dataset('endfoot_surface_coordinates', data=self.surface_coordinates)
            fd.create_dataset('endfoot_graph_coordinates', data=self.graph_coordinates)

    def test(self):

        with GliovascularData(self.filename) as gv_data:

            assert np.allclose(gv_data.endfoot_surface_coordinates, self.surface_coordinates)
            assert np.allclose(gv_data.endfoot_graph_coordinates, self.graph_coordinates)


###########################################################################################################
###########################################################################################################
###########################################################################################################


class MockSynapticData(object):

    def __init__(self, filename, n_synapses):

        self.filename = filename
        self.n_synapses = n_synapses

    @cached_property
    def synapse_coordinates(self):
        vals = np.arange(self.n_synapses, dtype=np.float)
        return np.column_stack((vals, vals, vals))

    def export(self):

        with h5py.File(self.filename, 'w') as fd:
            fd.create_dataset('synapse_coordinates', data=self.synapse_coordinates)

    def test(self):

        with SynapticData(self.filename) as syn_data:

            assert syn_data.n_synapses == self.n_synapses
            assert np.allclose(syn_data.synapse_positions, self.synapse_coordinates)


###########################################################################################################
###########################################################################################################
###########################################################################################################


class MockNeuroglialConnectivity(object):

    def __init__(self, filename, n_astrocytes, n_synapses, n_neurons):
        self.filename = filename
        self.n_astrocytes = n_astrocytes
        self.n_synapses = n_synapses
        self.n_neurons = n_neurons

    @cached_property
    def synapses_per_astrocyte(self):
        synapses = np.arange(self.n_synapses, dtype=np.uintp)
        return np.split(synapses, self.n_astrocytes)

    @cached_property
    def neurons_per_astrocyte(self):
        neurons = np.arange(self.n_neurons, dtype=np.uintp)
        return np.split(neurons, self.n_astrocytes)

    @cached_property
    def astrocytes_per_neuron(self):

        a_p_n = [[] for _ in range(self.n_neurons)]

        for astro_index, neurons in enumerate(self.neurons_per_astrocyte):
            for neuron in neurons:
                a_p_n[neuron].append(astro_index)
        return a_p_n

    @cached_property
    def neuron_to_astrocyte(self):
        res = np.zeros(self.n_synapses, dtype=np.uintp)
        offset = 0
        for astrocyte_index, neurons in enumerate(self.neurons_per_astrocyte):
            n_neurons = len(neurons)
            res[offset: offset + n_neurons] = astrocyte_index
            offset += n_neurons
        return res

    def data_generator(self, n_astrocytes):

        for synapses, neurons in zip(self.synapses_per_astrocyte, self.neurons_per_astrocyte):
            yield {'domain_synapses': synapses, 'domain_neurons':  neurons}

    def export(self):

        data_iterator = self.data_generator(self.n_astrocytes)
        n_astrocytes = self.n_astrocytes
        n_synapses = self.n_synapses
        n_neurons = self.n_neurons

        with h5py.File(self.filename, 'w') as fd:

            # ASTROCYTE PERSPECTIVE
            ########################################################################################################

            astrocyte_group = fd.create_group('/Astrocyte')
            astrocyte_offsets_dset = astrocyte_group.create_dataset('offsets', shape=(n_astrocytes + 1, 2), dtype=np.uintp)
            astrocyte_offsets_dset.attrs['column_names'] = np.array(['synapse', 'neuron'], dtype=h5py.special_dtype(vlen=str))

            astrocyte_synapse_dset = \
            astrocyte_group.create_dataset('synapse', shape=(n_synapses,), dtype=np.uintp)

            astrocyte_neuron_dset = \
            astrocyte_group.create_dataset('neuron', shape=(n_neurons,), dtype=np.uintp)

            synapse_offset = 0
            neuron_offset  = 0
            for index, domain_data in enumerate(data_iterator):

                synapse_indices = domain_data['domain_synapses']

                N = len(synapse_indices)

                astrocyte_synapse_dset[synapse_offset: synapse_offset + N] = synapse_indices

                synapse_offset += N

                ####################################################

                neuronal_indices = domain_data['domain_neurons']

                M = len(neuronal_indices)

                astrocyte_neuron_dset[neuron_offset: neuron_offset + M] = neuronal_indices

                neuron_offset += M

                ####################################################

                astrocyte_offsets_dset[index + 1, 0] = synapse_offset
                astrocyte_offsets_dset[index + 1, 1] = neuron_offset

            assert synapse_offset == len(astrocyte_synapse_dset)
            assert neuron_offset == len(astrocyte_neuron_dset)

            # NEURON PERSPECTIVE
            ########################################################################################################

            neuron_group = fd.create_group('/Neuron')
            neuron_offsets_dset = neuron_group.create_dataset('offsets', shape=(n_neurons + 1,), dtype=np.uintp)
            neuron_offsets_dset.attrs['column_names'] = np.array(['astrocyte'], dtype=h5py.special_dtype(vlen=str))
            neuron_offsets_dset[0] = 0

            neuron_astrocyte_dset = neuron_group.create_dataset('astrocyte', shape=(n_astrocytes,), dtype=np.uintp)

            offset = 0
            for neuron_index, astrocytes in enumerate(self.astrocytes_per_neuron):

                N = len(astrocytes)

                neuron_astrocyte_dset[offset: (offset + N)] = astrocytes

                offset += N

                neuron_offsets_dset[neuron_index + 1] = offset


    def test(self):

        synapses_per_astrocyte = self.synapses_per_astrocyte
        neurons_per_astrocyte = self.neurons_per_astrocyte

        with NeuroglialConnectivity(self.filename) as ng_conn:

            assert ng_conn.n_astrocytes == self.n_astrocytes

            for astro_index in range(self.n_astrocytes):
                assert np.allclose(ng_conn.astrocyte.to_synapse(astro_index), synapses_per_astrocyte[astro_index])
                assert np.allclose(ng_conn.astrocyte.to_neuron(astro_index), neurons_per_astrocyte[astro_index])

###########################################################################################################
###########################################################################################################
###########################################################################################################


class MockSynapticConnectivity(object):

    def __init__(self, filename, n_neurons, n_synapses):
        self.filename = filename
        self.n_neurons = n_neurons
        self.n_synapses = n_synapses

    @cached_property
    def synapses_per_neuron(self):
        synapses = np.arange(self.n_synapses, dtype=np.uintp)
        return np.split(synapses, self.n_neurons)

    @cached_property
    def synapse_to_neuron_connectivity(self):
        res = np.zeros(len(self), dtype=np.uintp)
        k = 0
        for neuron_index, neuron_synapses in enumerate(self.synapses_per_neuron):
            for _ in range(neuron_synapses.size):
                res[k] = neuron_index
                k += 1
        return res

    def export(self):

        n_neurons = self.n_neurons

        with h5py.File(self.filename, 'w') as fd_conn:

            # Synapse Point of view Group

            synapse_group = fd_conn.create_group('Synapse')
            dset_afferent_neuron = synapse_group.create_dataset('Afferent Neuron', data=self.synapse_to_neuron_connectivity)

            # Afferent Neuron Point of view Group

            afferent_neuron_group = fd_conn.create_group('Afferent Neuron')

            dset_afferent_neuron_offsets = \
            afferent_neuron_group.create_dataset('offsets',
                                                 shape=(n_neurons + 1,),
                                                 dtype='f16', chunks=None)
            dset_afferent_neuron_offsets[0] = 0
            dset_afferent_neuron_offsets.attrs['column_names'] = np.array(['Synapse'], dtype=h5py.special_dtype(vlen=str))

            offset = 0
            for neuron_index, synapses in enumerate(self.synapses_per_neuron):

                n_synapses =  len(synapses)
                dset_afferent_neuron[offset: offset + n_synapses] = neuron_index

                offset += n_synapses
                dset_afferent_neuron_offsets[neuron_index + 1] = offset


    def test(self):

        synapses_per_neuron = self.synapses_per_neuron

        with SynapticConnectivity(self.filename) as syn_conn:

            assert syn_conn.n_synapses == len(self)

            for neuron_index in range(self.n_neurons):

                assert np.allclose(syn_conn.afferent_neuron.to_synapse(neuron_index),
                                   synapses_per_neuron[neuron_index])

            s2n_conn = self.synapse_to_neuron_connectivity

            for synapse_index in range(self.n_synapses):

                assert syn_conn.synapse.to_afferent_neuron(synapse_index) == s2n_conn[synapse_index]
                

###########################################################################################################
###########################################################################################################
###########################################################################################################


class MockEnfeetome(object):

    def __init__(self, filename, n_astrocytes, n_endfeet):
        self.filename = filename
        self.n_endfeet = n_endfeet
        self.n_astrocytes = n_astrocytes

    def endfoot_points(self, endfoot_index):
        vals = np.arange(6)
        return np.column_stack((vals, vals, vals)) + float(endfoot_index)

    def endfoot_triangles(self, endfoot_index):
        return [[0, 1, 2],
                [2, 3, 4],
                [4, 5, 0]]

    @cached_property
    def endfeet_per_astrocyte(self):
        endfeet = np.arange(self.n_endfeet, dtype=np.uintp)
        return np.split(endfeet, self.n_astrocytes)

    @cached_property
    def endfoot_to_astrocyte(self):
        res = np.zeros(self.n_endfeet, dtype=np.uintp)
        k = 0
        for astro_index, endfeet in enumerate(self.endfeet_per_astrocyte):
            n_endfeet = len(endfeet)
            res[k: k + n_endfeet] = astro_index

            k += n_endfeet
        return res

    def data_generator(self, n_endfeet):

        astrocyte_indices = np.repeat(np.arange(n_endfeet / 2), 2)
        e2a = self.endfoot_to_astrocyte
        for i in range(n_endfeet):
            yield i, e2a[i], self.endfoot_points(i), self.endfoot_triangles(i)

    def export(self):

        data = self.data_generator(n_endfeet)

        with h5py.File(self.filename, 'w') as fd:

            metadata = fd.create_group('metadata')

            metadata.attrs['object_type'] = np.array(['endfoot_mesh'], dtype=h5py.special_dtype(vlen=str))

            meshes = fd.create_group('objects')

            for index, astrocyte_index, points, triangles in data:

                mesh_group = meshes.create_group('endfoot_{}'.format(index))

                mesh_group.create_dataset('points', data=points)
                mesh_group.create_dataset('triangles', data=triangles)

                mesh_group.attrs['astrocyte_index'] = astrocyte_index

    def test(self):

        e2a = self.endfoot_to_astrocyte

        with h5py.File(self.filename, 'r') as endf:

            meshes_group = endf['objects']

            n_endfeet = len(meshes_group)

            assert n_endfeet == self.n_endfeet

            for i in range(self.n_endfeet):

                endfoot_key = 'endfoot_{}'.format(i)

                endfoot_group = meshes_group[endfoot_key]

                assert np.allclose(endfoot_group['points'], self.endfoot_points(i))
                assert np.allclose(endfoot_group['triangles'], self.endfoot_triangles(i))
                assert endfoot_group.attrs['astrocyte_index'] == e2a[i]


###########################################################################################################
###########################################################################################################
###########################################################################################################


def realistic_numbers(n_cells):

    n_endfeet = 2 * n_cells
    n_neurons = 10 * n_cells
    n_synapses = 1500 * n_neurons

    assert n_cells < n_endfeet < n_neurons < n_synapses

    return n_endfeet, n_neurons, n_synapses

def toy_numbers(n_cells):
    return 8, 20, 40

if __name__ == '__main__':

    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = '.'

    #n_cells = 4
    n_cells = 10000
    n_endfeet, n_neurons, n_synapses = realistic_numbers(n_cells)
    #n_endfeet, n_neurons, n_synapses = toy_numbers(n_cells)

    L.info('Cell data creation started.')

    cell_data_filepath = os.path.join(output_path, 'cell_data.h5')
    mock_cell_data = MockCellData(cell_data_filepath, n_cells)
    mock_cell_data.export()
    mock_cell_data.test()
    del mock_cell_data

    L.info('Cell data creation completed.')
    ########################################################################################

    L.info('Microdomains started.')

    microdomains_filepath = os.path.join(output_path, 'microdomain_structure.h5')
    mock_microdomains = MockMicrodomainTesselation(microdomains_filepath, n_cells)
    mock_microdomains.export()
    mock_microdomains.test()
    del mock_microdomains

    L.info('Microdomains completed.')
    ########################################################################################

    L.info('Gliovascular Connectivity started.')

    gv_conn_filepath = os.path.join(output_path, 'gliovascular_connectivity.h5')
    mock_gliovascular_connectivity = MockGliovascularConnectivity(gv_conn_filepath, n_cells, n_endfeet)
    mock_gliovascular_connectivity.export()
    mock_gliovascular_connectivity.test()
    del mock_gliovascular_connectivity

    L.info('Gliovascular Connectivity completed.')

    ########################################################################################

    L.info('Gliovascular data started.')

    gv_data_filepath = os.path.join(output_path, 'gliovascular_data.h5')
    mock_gliovascular_data = MockGliovascularData(gv_data_filepath, n_cells, n_endfeet)
    mock_gliovascular_data.export()
    mock_gliovascular_data.test()
    del mock_gliovascular_data

    L.info('Gliovascular data completed.')

    ########################################################################################

    L.info('Synaptic data started.')

    synaptic_data_filepath = os.path.join(output_path, 'synaptic_data.h5')
    mock_synaptic_data = MockSynapticData(synaptic_data_filepath, n_synapses)
    mock_synaptic_data.export()
    mock_synaptic_data.test()
    del mock_synaptic_data

    L.info('Synaptic data completed.')

    ########################################################################################

    L.info('Synaptic Connectivity started.')

    syn_conn_filepath = os.path.join(output_path, 'synaptic_connectivity_test.h5')
    mock_synaptic_connectivity = MockSynapticConnectivity(syn_conn_filepath, n_neurons, n_synapses)
    mock_synaptic_connectivity.export()
    mock_synaptic_connectivity.test()
    del mock_synaptic_connectivity

    L.info('Synaptic Connectivity completed.')

    ########################################################################################

    L.info('Neuroglial Connectivity started.')

    ng_conn_filepath = os.path.join(output_path, 'neuroglial_connectivity_test.h5')
    mock_neuroglial_connectivity = MockNeuroglialConnectivity(ng_conn_filepath, n_cells, n_synapses, n_neurons)
    mock_neuroglial_connectivity.export()
    mock_neuroglial_connectivity.test()
    del mock_neuroglial_connectivity

    L.info('Neuroglial Connectivity completed.')

    #######################################################################################

    L.info('Endfeetome started.')

    endfeetome_filepath = os.path.join(output_path, 'endfeetome_test.h5')
    mock_endfeetome = MockEnfeetome(endfeetome_filepath, n_cells, n_endfeet)
    mock_endfeetome.export()
    mock_endfeetome.test()
    del mock_endfeetome

    L.info('Endfeetome completed.')
