from .context import multiple_view_geometry

import unittest
import numpy as np

from multiple_view_geometry.cube import Cube
from multiple_view_geometry.camera import Camera
from multiple_view_geometry.homogeneous_matrix import HomogeneousMatrix
from multiple_view_geometry.transform_utils import create_rotation_mat_from_rpy, points_to_homogeneous_coordinates
from multiple_view_geometry.algorithm import solve_essential_matrix


class SceneFixture(unittest.TestCase):
    def setUp(self):
        camera0_extrinsic = HomogeneousMatrix.create(
            [1.7, 0.0, 0.5], create_rotation_mat_from_rpy(-np.pi / 2, 0, -np.pi / 4)
        )
        self._camera0 = Camera("0", camera0_extrinsic)
        camera1_extrinsic = HomogeneousMatrix.create([2.3, 0.0, 0.5], create_rotation_mat_from_rpy(-np.pi / 2, 0.0, 0))
        self._camera1 = Camera("1", camera1_extrinsic)
        self._cube = Cube((2, 3, 0), (2, 2, 2), resolution=1)
        self._key_points_cube = self._cube.surfaces()
        _, self._points_in_image_frame0 = self._camera0.project(self._key_points_cube)
        _, self._points_in_image_frame1 = self._camera1.project(self._key_points_cube)


def add_gaussian_noise(points, mean, variance):
    noise = np.random.normal(mean, variance, points.shape)
    return points + noise


def add_gaussian_noise_to_homogeneous_matrix(
    mat, rotation_mean, rotation_variance, translation_mean, translation_variance
):
    noisy_rotation = add_gaussian_noise(mat.rotation, rotation_mean, rotation_variance)
    noisy_translation = add_gaussian_noise(mat.translation, translation_mean, translation_variance)
    return HomogeneousMatrix.create(noisy_translation, noisy_rotation)


class SceneWithNoiseFixture(SceneFixture):
    def setUp(self):
        SceneFixture.setUp(self)
        np.random.seed(1)

    def add_noise_in_pixel(self, mean, variance):
        self._points_in_image_frame0 = add_gaussian_noise(self._points_in_image_frame0, mean, variance)
        self._points_in_image_frame1 = add_gaussian_noise(self._points_in_image_frame1, mean, variance)

    def add_noise_in_3d_points(self, mean, variance):
        return add_gaussian_noise(self._key_points_cube, mean, variance)
