import numpy as np
import matplotlib.pyplot as plt

from scene import Scene
from cube import Cube
from camera import Camera
from camera_image_renderer import CameraImageRenderer
from homogeneous_matrix import HomogeneousMatrix
from transform_utils import create_rotation_mat_from_rpy

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

