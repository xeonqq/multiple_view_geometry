import numpy as np

class Cube(object):
    def __init__(self, base_point, dimensions, resolution):
        self._base_point = base_point
        self._dimensions = dimensions
        self._resolution = resolution

    def _surfaces(self, base_point, dimensions):
        x0, y0, z0 = base_point
        dim_x, dim_y, dim_z = dimensions
        x, y, z = np.meshgrid(np.arange(x0,x0+dim_x+self._resolution, self._resolution),
                np.arange(y0,y0+dim_y+self._resolution, self._resolution),
                np.arange(z0,z0+dim_z+self._resolution, self._resolution))
        key_points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
        return key_points

    def surfaces(self):
        dim_x, dim_y, dim_z = self._dimensions
        x, y ,z = self._base_point
        surface_xz = self._surfaces(self._base_point, (dim_x, 0, dim_z))
        surface_zy = self._surfaces((x, y+1, z), (0, dim_y-1, dim_z))
        key_points = np.hstack((surface_xz, surface_zy))
        return key_points



