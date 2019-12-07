from .context import multiple_view_geometry

import numpy as np
import g2o

from .scene_fixture import SceneWithNoiseFixture, add_gaussian_noise_to_homogeneous_matrix
from multiple_view_geometry.algorithm import (
    solve_essential_matrix,
    reconstruct_translation_and_rotation_from_svd_of_essential_matrix,
    structure_from_motion_options,
)
from multiple_view_geometry.homogeneous_matrix import HomogeneousMatrix
from multiple_view_geometry.bundle_ajustment import BundleAjustment


class TestBundleAdjustment(SceneWithNoiseFixture):
    def compute_transform_cam0_wrt_cam1_by_bundle_adjustment(
        self, transform_cam0_wrt_cam1, points_3d_in_camera_frame0, iterations=10, verbose=False
    ):

        bundle_adjustment = BundleAjustment()
        bundle_adjustment.add_camera_parameters(self._camera0.focal_length_in_pixels, self._camera0.pixel_center)
        # First camera frame0 is regarded as the origin of the world
        bundle_adjustment.add_pose(0, g2o.SE3Quat(R=np.identity(3), t=np.zeros(3)), fixed=True)
        # IMPORTANT NOTE: rotation and translation of pose is described as world wrt to pose
        bundle_adjustment.add_pose(
            1, g2o.SE3Quat(R=transform_cam0_wrt_cam1.rotation, t=transform_cam0_wrt_cam1.translation)
        )

        points_id = np.arange(0, len(points_3d_in_camera_frame0.T)) + 2

        for point_id, point_3d in zip(points_id, points_3d_in_camera_frame0.T):
            bundle_adjustment.add_point(point_id, point_3d)

        pose_id_to_points_map = {
            0: (points_id, self._points_in_image_frame0.T),
            1: (points_id, self._points_in_image_frame1.T),
        }
        for pose_id, bundle in pose_id_to_points_map.items():
            for point_id, measured_point_2d in zip(*bundle):
                bundle_adjustment.add_edge(point_id, pose_id, measured_point_2d, information=np.identity(2))

        bundle_adjustment.optimize(iterations, verbose)

        return HomogeneousMatrix(bundle_adjustment.vertex_estimate(1).matrix()[:3, :4])

    def test_bundle_adjustment_when_initial_condition_given_by_eight_point_algorithm(self):
        self.add_noise_in_pixel(0, 0.0)
        self.assertGreaterEqual(self._key_points_cube.shape[1], 8)
        # we know the points are not all on the same surface
        # and we use more than 8 points here to guarentee that rank of A is larger than 8
        essential_matrix, u, s, vh = solve_essential_matrix(
            self._points_in_image_frame0, self._camera0, self._points_in_image_frame1, self._camera1
        )

        # there are two possible solutions, after we implement reconstuct stuction function, we can check
        # the plausible solution should have positive depth
        T1, R1, T2, R2 = reconstruct_translation_and_rotation_from_svd_of_essential_matrix(u, s, vh)
        homogeneous_points_in_camera_frame0 = self._camera0.points_2d_to_homogeneous_coordinates(
            self._points_in_image_frame0
        )
        homegeneous_points_in_camera_frame1 = self._camera1.points_2d_to_homogeneous_coordinates(
            self._points_in_image_frame1
        )
        transform_cam1_wrt_cam0, points_depth_scale_in_cam1 = structure_from_motion_options(
            homogeneous_points_in_camera_frame0,
            homegeneous_points_in_camera_frame1,
            HomogeneousMatrix.create(T1, R1),
            HomogeneousMatrix.create(T2, R2),
        )

        points_3d_in_camera_frame1 = points_depth_scale_in_cam1 * homegeneous_points_in_camera_frame1
        points_3d_in_camera_frame0 = (
            transform_cam1_wrt_cam0.rotation.dot(points_3d_in_camera_frame1)
            + transform_cam1_wrt_cam0.translation[:, np.newaxis]
        )

        transform_cam0_wrt_cam1 = HomogeneousMatrix(transform_cam1_wrt_cam0.inv())
        transform_cam0_wrt_cam1_bundle_adjustment = self.compute_transform_cam0_wrt_cam1_by_bundle_adjustment(
            transform_cam0_wrt_cam1, points_3d_in_camera_frame0
        )

        # test the recovered translation and rotation is the same as the ground truth
        scale = 3.0 / 5
        ground_truth_transform = self._camera0.get_transform_wrt(self._camera1)
        # rotation and translation estimated by bundle adjustment
        np.testing.assert_array_almost_equal(
            transform_cam0_wrt_cam1_bundle_adjustment.translation * scale, ground_truth_transform.translation
        )
        np.testing.assert_array_almost_equal(
            transform_cam0_wrt_cam1_bundle_adjustment.rotation, ground_truth_transform.rotation
        )

    def test_bundle_adjustment_when_initial_condition_given_by_groundtruth_with_noise(self):
        self.add_noise_in_pixel(0, 2)
        self.add_noise_in_3d_points(0, 0.2)

        gt_transform_cam0_wrt_cam1 = self._camera0.get_transform_wrt(self._camera1)
        noisy_transform_cam0_wrt_cam1 = add_gaussian_noise_to_homogeneous_matrix(
            gt_transform_cam0_wrt_cam1, 0, 0.05, 0, 0.2
        )
        noisy_3d_points_in_camera0 = self._camera0.world_frame_to_camera_frame(self._key_points_cube)
        transform_cam0_wrt_cam1_bundle_adjustment = self.compute_transform_cam0_wrt_cam1_by_bundle_adjustment(
            noisy_transform_cam0_wrt_cam1, noisy_3d_points_in_camera0
        )

        np.testing.assert_array_almost_equal(
            transform_cam0_wrt_cam1_bundle_adjustment.translation, gt_transform_cam0_wrt_cam1.translation, decimal=0.1
        )
        np.testing.assert_array_almost_equal(
            transform_cam0_wrt_cam1_bundle_adjustment.rotation, gt_transform_cam0_wrt_cam1.rotation, decimal=0.1
        )
