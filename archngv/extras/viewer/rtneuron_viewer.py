import numpy

from matplotlib.colors import to_rgba
import rtneuron


def _concatenate_connectivity(points1, points2, connectivity_edges):
        nedges = connectivity_edges.copy()

        n_p1 = points1.shape[0]

        points = numpy.vstack((points1, points2))
        nedges[:, 1] += n_p1

        return points, nedges


def _pdata(points, radii):
    try:
        data = numpy.column_stack((points, radii))
    except ValueError:
        data = numpy.column_stack((points, radii * numpy.ones(len(points))))

    return numpy.ascontiguousarray(data)


class RTNeuronViewer(object):

    def __init__(self, view=None):
        if view is None:
            self.view = rtneuron.display_empty_scene()
        else:
            self.view = view
        self._as_spheres = rtneuron.AttributeMap({'point_style': 'spheres'})

    def add_lines(self, points, edges, color='w'):
        scene = self.view.scene
        scene.addGeometry(points, edges, colors=to_rgba(color))

    def add_mesh(self, filepath):
        self.view.scene.addModel(str(filepath))

    def add_point_data(self, pdata, color='g'):
        scene = self.view.scene
        scene.addGeometry(pdata, attributes=self._as_spheres, colors=to_rgba(color))

    def add_spheres(self, positions, radii, color='g'):
        self.add_point_data(_pdata(positions, radii), color=color)

    def add_bijective_mapping(self, points1, points2, radius=1., colors=('g', 'r', 'w')):

        assert len(points1) == len(points2)

        n_points = len(points1)

        radii = numpy.ones(n_points, dtype=numpy.float)

        idx = numpy.arange(n_points, dtype=numpy.uintp)
        connectivity_edges = numpy.column_stack((idx, idx))

        self.add_connectivity(points1, radii, points2, radii, connectivity_edges)


    def add_connectivity(self, points1, radii1, points2, radii2, connectivity_edges, colors=('g', 'r', 'w')):

        points, edges = \
        _concatenate_connectivity(points1, points2, connectivity_edges)

        self.add_lines(points, edges, color=colors[2])

        pdata1 = _pdata(points1, radii1)
        pdata2 = _pdata(points2, radii2)

        self.add_point_data(pdata1, color=colors[0])
        self.add_point_data(pdata2, color=colors[1])

    def add_bounding_box(self, low, hgh, color='w'):
        scene = self.view.scene
        xmin, ymin, zmin = low
        xmax, ymax, zmax = hgh

        points = numpy.array([[xmin, ymin, zmin],
                              [xmax, ymin, zmin],
                              [xmax, ymax, zmin],
                              [xmin, ymax, zmin],
                              [xmin, ymax, zmax],
                              [xmin, ymin, zmax],
                              [xmax, ymin, zmax],
                              [xmax, ymax, zmax]])

        edges = numpy.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [3, 0],
                             [3, 4],
                             [4, 5],
                             [5, 0],
                             [5, 6],
                             [6, 7],
                             [7, 4],
                             [1, 6],
                             [2, 7]])

        self.add_lines(points, edges, color=color)
