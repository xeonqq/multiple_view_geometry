import unittest
import numpy as np

from cube import Cube
from camera import Camera
from homogeneous_matrix import HomogeneousMatrix
from transform_utils import create_rotation_mat_from_rpy, points_to_homogeneous_coordinates, \
        calculate_direction_vecs_in_world_frame, triangulate


class TestTriangulation(unittest.TestCase):
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

    def test_triangulate_function(self):
        # test case comes from:
        # https://math.stackexchange.com/questions/270767/find-intersection-of-two-3d-lines
        start_points0 = np.array([5, 5, 4])
        dir_vec0 = np.array([10, 10, 6]) - start_points0
        start_points1 = np.array([5, 5, 5])
        dir_vec1 = np.array([10, 10, 3]) - start_points1
        point_3d = triangulate(start_points0, dir_vec0[:, np.newaxis], \
                start_points1, dir_vec1[:, np.newaxis])
        np.testing.assert_array_equal(point_3d.ravel(), np.array([25/4.0, 25/4.0, 9/2.0]))

    def test_given_cooresponding_points_in_image_frame_and_camera_extrinsic_then_3d_point_in_world_can_be_triangulated(self):
        dir_vecs0 = calculate_direction_vecs_in_world_frame( \
                self._points_in_image_frame0, self._camera0)
        dir_vecs1 = calculate_direction_vecs_in_world_frame( \
                self._points_in_image_frame1, self._camera1)
        points_3d = triangulate(self._camera0.extrinsic.translation, dir_vecs0, \
                self._camera1.extrinsic.translation, dir_vecs1)
        np.testing.assert_array_almost_equal(points_3d, self._key_points_cube)

if __name__ == '__main__':
    unittest.main()
