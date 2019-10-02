from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

from camera import Camera
from homogeneous_matrix import HomogeneousMatrix
from transform_utils import normalize_homogeneous_coordinates, create_rotation_mat_from_rpy, create_cube_key_points, points_to_homogeneous_coordinates, translation_to_skew_symetric_mat

class Scene(object):
    def __init__(self):
        camera1_extrinsic = HomogeneousMatrix.create([2.2,0.5,0.5], create_rotation_mat_from_rpy(-np.pi/2, 0, -np.pi/4))
        self._camera1 = Camera(camera1_extrinsic)
        camera2_extrinsic = HomogeneousMatrix.create([2.7,0.0,0.5], create_rotation_mat_from_rpy(-np.pi/2, 0.0, 0))
        self._camera2 = Camera(camera2_extrinsic)
        self._key_points_cube = create_cube_key_points((2,3,0),(1,1,1), 1)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('keypoints in image frame')
        points_in_camera1, points_in_image_frame1= self._camera1.project(points_to_homogeneous_coordinates(self._key_points_cube), ax1, 'red')
        points_in_camera2, points_in_image_frame2 = self._camera2.project(points_to_homogeneous_coordinates(self._key_points_cube), ax2, 'blue')

        #print(self._camera1.extrinsic.mat)
        #print(self._camera1.extrinsic.inv())
        #print(self._camera2.extrinsic.mat)
        tf_cam2_wrt_cam1 = self._camera1.extrinsic.inv().dot(self._camera2.extrinsic.mat) # the translation from it is not very correct
        #tf_cam1_wrt_cam2 = self._camera2.extrinsic.inv().dot(self._camera1.extrinsic.mat)

        #print (tf_cam2_wrt_cam1)
        #print (tf_cam1_wrt_cam2)

        #hack:
        #tf_cam2_wrt_cam1[:3,3] = -tf_cam1_wrt_cam2[:3,3]

        R_cam2_wrt_cam1 = tf_cam2_wrt_cam1[:3,:3]
        T_cam2_wrt_cam1 = tf_cam2_wrt_cam1[:3,3]
        print(points_in_camera2[:,0])
        print(translation_to_skew_symetric_mat(T_cam2_wrt_cam1))
        print(R_cam2_wrt_cam1)

        print(points_in_camera2)
        for p in points_in_camera2.T:
            epipolar_line_in_camera1 = translation_to_skew_symetric_mat(T_cam2_wrt_cam1).dot(R_cam2_wrt_cam1).dot(p)
            a, b, c = epipolar_line_in_camera1
            print (epipolar_line_in_camera1)
            x1, x2 = -2, 2
            line_start = np.array([x1, (-c+x1*a)/b, 1])
            line_start = normalize_homogeneous_coordinates(self._camera2.intrinsic[:,:3].dot(line_start))
            line_end = np.array([x2, (-c-x2*a)/b, 1])
            line_end = normalize_homogeneous_coordinates(self._camera2.intrinsic[:,:3].dot(line_end))
            line = np.vstack((line_start, line_end))
            ax1.plot(line[:,0], line[:,1])



if __name__ == '__main__':
    scene = Scene()
    plt.show()

