from .context import multiple_view_geometry

import numpy as np
import g2o

from .scene_fixture import SceneWithNoiseFixture
from multiple_view_geometry.algorithm import solve_essential_matrix, calculate_essential_matrix, reconstruct_translation_and_rotation_from_svd_of_essential_matrix, structure_from_motion_options
from multiple_view_geometry.homogeneous_matrix import HomogeneousMatrix
from multiple_view_geometry.bundle_ajustment import BundleAjustment


class TestBundleAdjustment(SceneWithNoiseFixture):
    def test_bundle_adjustment(self):
        self.assertGreaterEqual(self._key_points_cube.shape[1], 8)
        # we know the points are not all on the same surface
        # and we use more than 8 points here to guarentee that rank of A is larger than 8
        essential_matrix, u, s, vh = solve_essential_matrix(
                self._points_in_image_frame0, self._camera0,
                self._points_in_image_frame1, self._camera1)

        # there are two possible solutions, after we implement reconstuct stuction function, we can check
        # the plausible solution should have positive depth
        T1, R1, T2, R2 = reconstruct_translation_and_rotation_from_svd_of_essential_matrix(u, s, vh)
        homogeneous_points_in_camera_frame0 = self._camera0.points_2d_to_homogeneous_coordinates(self._points_in_image_frame0)
        homegeneous_points_in_camera_frame1 = self._camera1.points_2d_to_homogeneous_coordinates(self._points_in_image_frame1)
        transform_cam1_wrt_cam0, points_depth_scale_in_cam1 = structure_from_motion_options(
            homogeneous_points_in_camera_frame0,
            homegeneous_points_in_camera_frame1,
            HomogeneousMatrix.create(T1, R1), HomogeneousMatrix.create(T2, R2))

        points_3d_in_camera_frame1 = points_depth_scale_in_cam1 * homegeneous_points_in_camera_frame1
        points_3d_in_camera_frame0 = transform_cam1_wrt_cam0.rotation.dot(points_3d_in_camera_frame1) + \
                                     transform_cam1_wrt_cam0.translation[:, np.newaxis]

        bundle_adjustment = BundleAjustment()
        bundle_adjustment.add_camera_parameters(self._camera0.focal_length_in_pixels, self._camera0.pixel_center)
        bundle_adjustment.add_pose(0, g2o.SE3Quat(np.zeros(6)), fixed=True)
        bundle_adjustment.add_pose(1, g2o.SE3Quat(R=transform_cam1_wrt_cam0.rotation,
                                                  t=transform_cam1_wrt_cam0.translation))

        pose_id_to_points_map = {
            0: (points_3d_in_camera_frame0.T, self._points_in_image_frame0.T),
            1: (points_3d_in_camera_frame1.T, self._points_in_image_frame1.T)}

        point_id = 2
        for pose_id, (points_3d, measured_points_2d) in pose_id_to_points_map.items():
            for point_3d, measured_point_2d in zip(points_3d, measured_points_2d):
                bundle_adjustment.add_point(point_id, point_3d)
                bundle_adjustment.add_edge(point_id, pose_id, measured_point_2d, information=np.identity(2))
                point_id += 1

        bundle_adjustment.optimize()

        transform_cam1_wrt_cam0_bundle_adjustment = HomogeneousMatrix(bundle_adjustment.vertex_estimate(1).matrix()[:3,:4])

        # test the recovered translation and rotation is the same as the ground truth
        scale = 3./5
        ground_truth_transform = self._camera1.get_transform_wrt(self._camera0)
        # rotation and translation estimated by bundle adjustment
        # np.testing.assert_array_almost_equal(transform_cam1_wrt_cam0_bundle_adjustment.rotation, ground_truth_transform.rotation)

        # rotation and translation estimated by 8 point algorithm
        np.testing.assert_array_almost_equal(transform_cam1_wrt_cam0.rotation, ground_truth_transform.rotation)
        np.testing.assert_array_almost_equal(transform_cam1_wrt_cam0.translation*scale, ground_truth_transform.translation)
