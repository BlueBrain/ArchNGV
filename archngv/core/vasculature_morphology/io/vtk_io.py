import vtk
import logging
from vtk.util import numpy_support as _ns
import numpy as np
import warnings


log = logging.getLogger(__name__)


def vtk_points(points):
    """ Converts an array of numpy points to vtk points
    """
    vpoints = vtk.vtkPoints()

    vpoints.SetData(_ns.numpy_to_vtk(points.copy(), deep=1))


    return vpoints


def vtk_lines(edges):
    """ Converts a list of edges into vtk lines
    """
    vlines = vtk.vtkCellArray()

    n_edges = edges.shape[0]

    arr = np.empty((n_edges, 3), order='C', dtype=np.int)

    arr[:, 0] = 2 * np.ones(n_edges, dtype=np.int)
    arr[:, 1:] = edges

    # cell array structure: size of cell followed by edges
    #arr = np.column_stack((2 * np.ones(edges.shape[0], dtype=np.int), edges)).copy()

    # crucial to deep copy the data!!!
    vlines.SetCells(edges.shape[0],_ns.numpy_to_vtkIdTypeArray(arr, deep=1))
    return vlines


def vtk_attribute_array(name, arr):
    """ Creates a cell array with specified name and assignes the
    numpy array arr
    """
    val_arr = vtk.util.numpy_support.numpy_to_vtk(arr)
    val_arr.SetName(name)

    return val_arr


def create_polydata_from_data(points, edges, attribute_dict={}):
    """ Creates a PolyData vtk object from a set of points that are
    connected with edges and optionally have a set of attributes
    """
    polydata = vtk.vtkPolyData()

    polydata.SetPoints(vtk_points(points))
    polydata.SetLines(vtk_lines(edges))

    cell_data = polydata.GetCellData()

    for key, arr in attribute_dict.items():
        cell_data.AddArray(vtk_attribute_array(key, arr))

    return polydata


def vtk_loader(filename):
    """ Extracts from a vtk file the points, edges, radii and types
    """
    from vtk.util.numpy_support import vtk_to_numpy    

    def get_points(polydata):

        vtk_points = polydata.GetPoints()
        return vtk_to_numpy(vtk_points.GetData())

    def get_structure(polydata):

        vtk_lines = polydata.GetLines()

        nmp_lines = vtk_to_numpy(vtk_lines.GetData())

        return nmp_lines.reshape(len(nmp_lines)/3, 3)[:, (1, 2)].astype(np.intp)

    def get_radii(polydata):

        cell_data = polydata.GetCellData()

        N = cell_data.GetNumberOfArrays()

        names = [cell_data.GetArrayName(i) for i in range(N)]

        vtk_floats = cell_data.GetArray(names.index("radius"))

        return vtk_to_numpy(vtk_floats)

    def get_types(polydata):
        cell_data = polydata.GetCellData()

        N = cell_data.GetNumberOfArrays()

        names = [cell_data.GetArrayName(i) for i in range(N)]

        vtk_floats = cell_data.GetArray(names.index("types"))

        return vtk_to_numpy(vtk_floats)


    # craete a polydata reader
    reader = vtk.vtkPolyDataReader()

    # add the filename that will be read
    reader.SetFileName(filename)

    # update the output of the reader
    reader.Update()

    polydata = reader.GetOutput()

    points = get_points(polydata)
    edges = get_structure(polydata)
    radii = get_radii(polydata)

    # if no types are provided, it will return an array of zeros
    try:

        types = get_types(polydata)

    except:

        warnings.warn("Types were not found. Zeros are used instead.")
        types = np.zeros(edges.shape[0], dtype=np.int)

    return points, edges, radii, types


def vtk_writer(filename, points, edges, radii, types, mode='ascii'):
    """ Creates a vtk legacy file and populates it with a polydata
    object that is generated using the points, edges, radii and types
    """
    from .vtk_io import vtk_points, vtk_lines, vtk_attribute_array

    points = np.ascontiguousarray(points)
    edges = np.ascontiguousarray(edges)
    radii = np.ascontiguousarray(radii)
    types = np.ascontiguousarray(types)

    attr_dict = {'radius': radii, 'type': types}

    polydata = create_polydata_from_data(points, edges, attribute_dict=attr_dict)

    writer = vtk.vtkPolyDataWriter()

    writer.SetFileName(filename + '.vtk')

    if mode == "binary":
        writer.SetFileTypeToBinary()
    elif mode == "ascii":
        writer.SetFileTypeToASCII()

    writer.SetInputData(polydata)
    writer.Write()
