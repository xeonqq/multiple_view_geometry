from .context import multiple_view_geometry

import unittest
import numpy as np

from .scene_fixture import SceneFixture
from multiple_view_geometry.algorithm import solve_essential_matrix, calculate_essential_matrix, reconstruct_translation_and_rotation_from_svd_of_essential_matrix

class TestEightPointAlgorithm(SceneFixture):
    def test_eight_point_algorithm(self):
        self.assertGreaterEqual(self._key_points_cube.shape[1], 8)
        # we know the points are not all on the same surface
        # and we use more than 8 points here to guarentee that rank of A is 8
        essential_matrix, u, s, vh = solve_essential_matrix( \
                self._points_in_image_frame0, self._camera0, \
                self._points_in_image_frame1, self._camera1)
        true_essential_matrix = calculate_essential_matrix(self._camera1, self._camera0)

        #tf = self._camera1.get_transform_wrt(self._camera0)
        #T = tf.translation
        #R = tf.rotation

        # there are two possible solutions, after we implement reconstuct stuction function, we can check
        # the plausible solution should have positive depth
        # T1, R1, T2, R2 = reconstruct_translation_and_rotation_from_svd_of_essential_matrix(u, s, vh)

        scale = 3./5.
        # the calculated essential matrix is the same as the true_essential_matrix by a scale
        np.testing.assert_array_almost_equal(essential_matrix*(scale), true_essential_matrix)


if __name__ == '__main__':
    unittest.main()
