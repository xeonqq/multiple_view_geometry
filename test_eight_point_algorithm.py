import unittest
import numpy as np

from scene_fixture import SceneFixture
from transform_utils import solve_essential_matrix, calculate_essential_matrix

class TestEightPointAlgorithm(SceneFixture):
    def test_eight_point_algorithm(self):
        self.assertGreaterEqual(self._key_points_cube.shape[1], 8)
        # we know the points are not all on the same surface
        # and we use more than 8 points here to guarentee that rank of A is 8
        essential_matrix = solve_essential_matrix( \
                self._points_in_image_frame0, self._camera0, \
                self._points_in_image_frame1, self._camera1)
        true_essential_matrix = calculate_essential_matrix(self._camera1, self._camera0)

        # the calculated essential matrix is the same as the true_essential_matrix by a scale
        np.testing.assert_array_almost_equal(essential_matrix/(5./3.), true_essential_matrix)

if __name__ == '__main__':
    unittest.main()
