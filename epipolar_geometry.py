import numpy as np
import matplotlib.pyplot as plt

from cube import Cube
from camera import Camera
from camera_image_renderer import CameraImageRenderer
from homogeneous_matrix import HomogeneousMatrix
from linear_equation import LinearEquation
from transform_utils import normalize_homogeneous_coordinates, create_rotation_mat_from_rpy, points_to_homogeneous_coordinates, translation_to_skew_symetric_mat, calculate_epipolar_line_on_other_image, calculate_essential_matrix

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

        essential_matrix_cam1 = calculate_essential_matrix(self._cameras[1], self._cameras[0])

        for p in lists_of_points_in_camera_frame[1].T:
            line = calculate_epipolar_line_on_other_image(p, essential_matrix_cam1, self._cameras[0])
            self._renderer.render_epipolar_line(line, self._cameras[0])

        essential_matrix_cam0 = calculate_essential_matrix(self._cameras[0], self._cameras[1])
        for p in lists_of_points_in_camera_frame[0].T:
            line = calculate_epipolar_line_on_other_image(p, essential_matrix_cam0, self._cameras[1])
            self._renderer.render_epipolar_line(line, self._cameras[1])


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

