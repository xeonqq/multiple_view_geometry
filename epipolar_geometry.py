from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

from cube import Cube
from camera import Camera
from homogeneous_matrix import HomogeneousMatrix
from linear_equation import LinearEquation
from transform_utils import normalize_homogeneous_coordinates, create_rotation_mat_from_rpy, points_to_homogeneous_coordinates, translation_to_skew_symetric_mat

class Scene(object):
    def __init__(self, cube, cameras=[]):
        self._cube = cube
        self._cameras = cameras

    def project(self, colors=[], show_plot=True):
        assert len(colors) == len(self._cameras)
        fig, axes = plt.subplots(1, len(self._cameras))
        fig.suptitle('keypoints in image frame')

        lists_of_points_in_camera_frame = []
        lists_of_points_in_image_frame = []
        key_points_cube = cube.surfaces()
        homogeneous_key_points_cube = points_to_homogeneous_coordinates(key_points_cube)
        for camera, color, ax in zip(self._cameras, colors, axes):
            points_in_camera_frame, points_in_image_frame = camera.project(homogeneous_key_points_cube, ax, color, show_plot)
            lists_of_points_in_camera_frame.append(points_in_camera_frame)
            lists_of_points_in_image_frame.append(points_in_image_frame)

        # tf_cam1_wrt_cam0 = tf_world_wrt_cam0 * tf_cam1_wrt_world
        tf_cam1_wrt_cam0 = self._cameras[0].extrinsic.inv().dot(self._cameras[1].extrinsic.mat)

        R_cam1_wrt_cam0 = tf_cam1_wrt_cam0[:3,:3]
        T_cam1_wrt_cam0 = tf_cam1_wrt_cam0[:3,3]

        # because of epipolar geometry
        # p0*E*p1 = 0
        # Essential matrix = T_cam1_wrt_cam0 cross multiply R_cam1_wrt_cam0
        # E = t x R
        essential_matrix = translation_to_skew_symetric_mat(T_cam1_wrt_cam0).dot(R_cam1_wrt_cam0)

        for p in lists_of_points_in_camera_frame[1].T:
            epipolar_line_in_camera0 = essential_matrix.dot(p)
            lineq = LinearEquation(epipolar_line_in_camera0)
            x1, x2 = -2, 2
            line_start = np.array([x1, lineq.solve_y(x1), 1])
            line_start = normalize_homogeneous_coordinates(self._cameras[0].intrinsic[:,:3].dot(line_start))
            line_end = np.array([x2, lineq.solve_y(x2), 1])
            line_end = normalize_homogeneous_coordinates(self._cameras[0].intrinsic[:,:3].dot(line_end))
            line = np.vstack((line_start, line_end))
            axes[0].plot(line[:,0], line[:,1])



if __name__ == '__main__':
    camera1_extrinsic = HomogeneousMatrix.create([1.7,0.0,0.5], create_rotation_mat_from_rpy(-np.pi/2, 0, -np.pi/4))
    camera1 = Camera(camera1_extrinsic)
    camera2_extrinsic = HomogeneousMatrix.create([2.3,0.0,0.5], create_rotation_mat_from_rpy(-np.pi/2, 0.0, 0))
    camera2 = Camera(camera2_extrinsic)

    cube = Cube((2,3,0), (2,2,2), resolution=1)
    scene = Scene(cube, [camera1, camera2])
    scene.project(['red', 'blue'])
    plt.show()

