import unittest
import numpy as np

from cube import Cube
from camera import Camera
from homogeneous_matrix import HomogeneousMatrix
from transform_utils import create_rotation_mat_from_rpy, points_to_homogeneous_coordinates

class SceneFixture(unittest.TestCase):
    def setUp(self):
        camera0_extrinsic = HomogeneousMatrix.create([1.7, 0.0, 0.5], \
                create_rotation_mat_from_rpy(-np.pi/2, 0, -np.pi/4))
        self._camera0 = Camera('0', camera0_extrinsic)
        camera1_extrinsic = HomogeneousMatrix.create([2.3, 0.0, 0.5], \
                create_rotation_mat_from_rpy(-np.pi/2, 0.0, 0))
        self._camera1 = Camera('1', camera1_extrinsic)
        self._cube = Cube((2, 3, 0), (1, 1, 1), resolution=1)
        self._key_points_cube = self._cube.surfaces()
        homogeneous_key_points_cube = points_to_homogeneous_coordinates(self._key_points_cube)
        _, self._points_in_image_frame0 = self._camera0.project(homogeneous_key_points_cube)
        _, self._points_in_image_frame1 = self._camera1.project(homogeneous_key_points_cube)
