import numpy as np
from .hexagonal_masking import mask_points_by_geometry

class NeuronalCircuitWrapper(object):

    @classmethod
    def from_geometry(cls, geo, center_tile=(1, 1)):

        points = np.asarray([x.vertices() for x in geo.mosaic_dict.values()])

        center_coords = geo.get(center_tile).center

        return cls(points, center_coords)

    @classmethod
    def test_microcircuit(cls):
        vertices = np.array([[[ 115.46,  599.94],
                           [ 230.92,  399.96],
                           [ 461.84,  399.96],
                           [ 577.3 ,  599.94],
                           [ 461.84,  799.92],
                           [ 230.92,  799.92]]])
        points = vertices.reshape(vertices.shape[0] * vertices.shape[1], vertices.shape[2])

        centers = np.array([[ 346.38,  599.94]])
        center = centers[0]

        layers = np.array([[ 2006.3482524,  1847.3347832,  1703.8656135,  1363.6375343,  1180.8844627,   674.6820627],
                           [ 1847.3347832,  1703.8656135,  1363.6375343,  1180.8844627, 674.6820627,     0.       ]])


        return cls(points, center, 0, 2006.3482524, centers, vertices, layers)


    @classmethod
    def test_mesocircuit(cls):

        vertices = np.array([[[ 115.46,  199.98], [ 230.92,    0.  ], [ 461.84,    0.  ], [ 577.3 ,  199.98], [ 461.84,  399.96], [ 230.92,  399.96]],
                                  [[ 461.84,  399.96], [ 577.3 ,  199.98], [ 808.22,  199.98], [ 923.68,  399.96], [ 808.22,  599.94], [ 577.3 ,  599.94]],
                                  [[  115.46,   999.9 ], [  230.92,   799.92], [  461.84,   799.92], [  577.3 ,   999.9 ], [  461.84,  1199.88], [  230.92,  1199.88]],
                                  [[-230.92,  799.92], [-115.46,  599.94], [ 115.46,  599.94], [ 230.92,  799.92], [ 115.46,  999.9 ], [-115.46,  999.9 ]],
                                  [[ 461.84,  799.92], [ 577.3 ,  599.94], [ 808.22,  599.94], [ 923.68,  799.92], [ 808.22,  999.9 ], [ 577.3 ,  999.9 ]],
                                  [[-230.92,  399.96], [-115.46,  199.98], [ 115.46,  199.98], [ 230.92,  399.96], [ 115.46,  599.94], [-115.46,  599.94]],
                                  [[ 115.46,  599.94], [ 230.92,  399.96], [ 461.84,  399.96], [ 577.3 ,  599.94], [ 461.84,  799.92], [ 230.92,  799.92]]])

        points = vertices.reshape(vertices.shape[0] * vertices.shape[1], vertices.shape[2])


        centers = np.array([[ 346.38,  199.98],
                                 [ 692.76,  399.96],
                                 [ 346.38,  999.9 ],
                                 [   0.  ,  799.92],
                                 [ 692.76,  799.92],
                                 [   0.  ,  399.96],
                                 [ 346.38,  599.94]])

        center = np.array([ 346.38,  599.94])

        layers = np.array([[ 2006.3482524,  1847.3347832,  1703.8656135,  1363.6375343,  1180.8844627,   674.6820627],
                           [ 1847.3347832,  1703.8656135,  1363.6375343,  1180.8844627, 674.6820627,     0.       ]])
        return cls(points, center, 0, 2006.3482524, centers, vertices, layers)

    def __init__(self, points, center, ymin, ymax, centers, vertices, layers):

        self._a1 = 230.92 
        self._a2 = 230.91776025243271
        self._ymin = ymin
        self._ymax = ymax
        self._angle = 2. * np.pi / 3.
        self._center = center
        self._vertices = points
        self.centers = centers
        self.vertices = vertices
        self.layers = layers

    @property
    def center(self):
        return self._center

    @property
    def base_length(self):
      tile = self.vertices[0, ...]
      return np.linalg.norm(tile[1, :] - tile[0, :])

    @property
    def volume(self):
      Dy = self._ymax - self._ymin

      return  self.mosaic_area * Dy

    @property
    def tile_height(self):
        return np.sin(self._angle)

    @property
    def height(self):
        return self._a2 * 2. * np.cos(self._angle - np.pi / 2.)

    @property
    def mosaic_area(self):

        a1 = self._a1
        a2 = self._a2
        N_tiles = self.centers.shape[0]
        l = self.height
        return l * (a1 + np.sin(self._angle - np.pi / 2.) * a1) * N_tiles

    @property
    def extent(self):
        x_ext = np.array([self.vertices[..., 0].min(), self.vertices[..., 0].max()])
        z_ext = np.array([self.vertices[..., 1].min(), self.vertices[..., 1].max()])
        y_ext = np.array([self._ymin, self._ymax])
        return x_ext, y_ext, z_ext

    @property
    def inscribed_extent(self):

        xe, ye, ze = self.extent

        s1 = np.cos((np.pi - self._angle)) * self._a
        s2 = np.sin((np.pi - self._angle)) * self._b

        return xe + np.array([s1, -s1]), ye, ze + np.array([s2, -s2])

    def tile_layout(self):

        V_N = self._vertices.shape[0]

        T_N = 6

        points = np.zeros(shape=(V_N * 2, 3))

        points[:, 0] = np.tile(self._vertices[:, 0], 2)
        points[:, 1] = np.hstack((np.repeat(self._ymin, V_N), np.repeat(self._ymax, V_N)))
        points[:, 2] = np.tile(self._vertices[:, 1], 2)

        edges = np.zeros(shape=( 3 * V_N, 2), dtype=np.intp)

        n = V_N / T_N

        t_inds = np.arange(T_N, dtype=np.intp)
        b_inds = np.repeat(np.arange(n, dtype=np.intp) * T_N , T_N)

        inds1 = np.tile(t_inds, n) + b_inds
        inds2 = np.tile(np.roll(t_inds, -1), n) + b_inds

        edges[:, 0] = np.hstack((inds1, inds1 + V_N, inds1))
        edges[:, 1] = np.hstack((inds2, inds2 + V_N, inds1 + V_N))

        return points, edges

    def mask_points(self, points):

      return mask_points_by_geometry(points, self)

