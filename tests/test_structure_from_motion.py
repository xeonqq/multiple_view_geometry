from .context import multiple_view_geometry

import unittest
import numpy as np

from .scene_fixture import SceneFixture
from multiple_view_geometry.algorithm import structure_from_motion

class TestStuctureFromMotion(SceneFixture):
    def test_structure_from_motion(self):
        self._points_in_image_frame0, self._camera0,
        self._points_in_image_frame1, self._camera1

        homogeneous_points_in_camera_frame0 = self._camera0.points_2d_to_homogeneous_coordinates(self._points_in_image_frame0)
        homogeneous_points_in_camera_frame1 = self._camera1.points_2d_to_homogeneous_coordinates(self._points_in_image_frame1)

        transform = self._camera1.get_transform_wrt(self._camera0)
        world_points_in_camera1 = structure_from_motion(homogeneous_points_in_camera_frame0, homogeneous_points_in_camera_frame1, transform);
        true_world_points_in_camera1 = self._camera1.world_frame_to_camera_frame(self._key_points_cube)
        np.testing.assert_array_almost_equal(world_points_in_camera1, true_world_points_in_camera1)

if __name__ == '__main__':
    unittest.main()
