from .context import multiple_view_geometry

import unittest
import numpy as np

from .scene_fixture import SceneFixture
from multiple_view_geometry.algorithm import triangulate, reconstruct_3d_points


class TestTriangulation(SceneFixture):
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
        points_3d = reconstruct_3d_points( \
                self._points_in_image_frame0, self._camera0, \
                self._points_in_image_frame1, self._camera1)
        np.testing.assert_array_almost_equal(points_3d, self._key_points_cube)

if __name__ == '__main__':
    unittest.main()
