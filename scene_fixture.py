import unittest
import numpy as np

from cube import Cube
from camera import Camera
from homogeneous_matrix import HomogeneousMatrix
from transform_utils import create_rotation_mat_from_rpy, points_to_homogeneous_coordinates
from algorithm import solve_essential_matrix


class SceneFixture(unittest.TestCase):
    def setUp(self):
        camera0_extrinsic = HomogeneousMatrix.create([1.7, 0.0, 0.5], \
                create_rotation_mat_from_rpy(-np.pi/2, 0, -np.pi/4))
        self._camera0 = Camera('0', camera0_extrinsic)
        camera1_extrinsic = HomogeneousMatrix.create([2.3, 0.0, 0.5], \
                create_rotation_mat_from_rpy(-np.pi/2, 0.0, 0))
        self._camera1 = Camera('1', camera1_extrinsic)
        self._cube = Cube((2, 3, 0), (2, 2, 2), resolution=1)
        self._key_points_cube = self._cube.surfaces()
        _, self._points_in_image_frame0 = self._camera0.project(self._key_points_cube)
        _, self._points_in_image_frame1 = self._camera1.project(self._key_points_cube)

