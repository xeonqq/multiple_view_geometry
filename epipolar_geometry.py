from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

from cube import Cube
from camera import Camera
from homogeneous_matrix import HomogeneousMatrix
from linear_equation import LinearEquation
from transform_utils import normalize_homogeneous_coordinates, create_rotation_mat_from_rpy, points_to_homogeneous_coordinates, translation_to_skew_symetric_mat

class CameraImageRenderer(object):
    def __init__(self, cameras_with_color, show_image_frame, show_epipolar_lines):
        self._show_image_frame = show_image_frame
        self._show_epipolar_lines = show_epipolar_lines
        self._fig, self._axes = plt.subplots(1, len(cameras_with_color))
        self._cameras_with_color = cameras_with_color
        self._fig.suptitle('keypoints in image frame')

    def render_image_frame_in_camera(self, points_in_image_frame, camera):
        if self._show_image_frame:
            self._axes[camera.id].set_xlim(0, camera.image_resolution[0])
            self._axes[camera.id].set_ylim(0, camera.image_resolution[1])
            self._axes[camera.id].set_title('image frame in camera {}'.format(camera.id))
            self._axes[camera.id].plot(points_in_image_frame[0,:], points_in_image_frame[1,:], 'o', color=self._cameras_with_color[camera])

    def render_epipolar_line(self, line_end_points, camera):
        if self._show_epipolar_lines:
            self._axes[camera.id].plot(line_end_points[:,0], line_end_points[:,1])
            plt.waitforbuttonpress(0.1)
            plt.draw()

class Scene(object):
    def __init__(self, cube, cameras, renderer):
        self._cube = cube
        self._cameras = cameras
        self._renderer = renderer

    def project(self, show_plot=True):
        lists_of_points_in_camera_frame = []
        lists_of_points_in_image_frame = []
        key_points_cube = cube.surfaces()
        homogeneous_key_points_cube = points_to_homogeneous_coordinates(key_points_cube)
        for camera in self._cameras:
            points_in_camera_frame, points_in_image_frame = camera.project(homogeneous_key_points_cube)
            lists_of_points_in_camera_frame.append(points_in_camera_frame)
            lists_of_points_in_image_frame.append(points_in_image_frame)
            self._renderer.render_image_frame_in_camera(points_in_image_frame, camera)

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
            self._renderer.render_epipolar_line(line, self._cameras[0])


if __name__ == '__main__':
    camera1_extrinsic = HomogeneousMatrix.create([1.7,0.0,0.5], create_rotation_mat_from_rpy(-np.pi/2, 0, -np.pi/4))
    camera1 = Camera(camera1_extrinsic)
    camera2_extrinsic = HomogeneousMatrix.create([2.3,0.0,0.5], create_rotation_mat_from_rpy(-np.pi/2, 0.0, 0))
    camera2 = Camera(camera2_extrinsic)

    cube = Cube((2,3,0), (2,2,2), resolution=1)
    renderer = CameraImageRenderer({camera1: 'red', camera2: 'blue'}, show_image_frame=True, show_epipolar_lines=True)
    scene = Scene(cube, [camera1, camera2], renderer)
    scene.project()
    plt.waitforbuttonpress(-1)

