from cv2 import cv2
import numpy as np
from math import *
import matplotlib.pyplot as plt

def create_rotation_mat_from_rpy(roll, pitch, yaw):
    Rx = np.array([
        1, 0, 0,
        0, cos(roll),  -sin(roll),
        0, sin(roll), cos(roll)]).reshape((3,3))
    Ry=np.array([
        cos(pitch), 0, sin(pitch),
        0 ,1, 0,
        -sin(pitch), 0, cos(pitch)]).reshape((3,3))
    Rz = np.array([
        cos(yaw), -sin(yaw), 0,
        sin(yaw), cos(yaw), 0,
        0, 0, 1]).reshape((3,3))
    R = Rz.dot(Ry.dot(Rx))
    return R

def create_cube_key_points(base_point, dimensions, resolution):
    # create 3 * width points in matrix form
    x0, y0, z0 = base_point
    dim_x, dim_y, dim_z = dimensions
    x, y, z = np.meshgrid(np.arange(x0,x0+dim_x+1, resolution),
            np.arange(y0,y0+dim_y+1, resolution),
            np.arange(z0,z0+dim_z+1, resolution))
    key_points = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    return key_points

def points_to_homogeneous_coordinates(points):
    dim, num = points.shape
    return np.vstack((points, np.ones((1, num))))

def normalize_homogeneous_coordinates(points):
    normalized_points = []
    if points.ndim == 2:
        normalized_points = points[:-1, :] / points[-1,:]
    elif points.ndim == 1:
        normalized_points = points[:-1] / points[-1]
    return normalized_points


def translation_to_skew_symetric_mat(translation):
    a1, a2, a3 = translation
    return np.array([[0, -a3, a2],[a3, 0, -a1],[-a2, a1, 0]])

class HomogeneousMatrix(object):
    def __init__(self, mat):
        self._mat = mat

    @staticmethod
    def create(translation, rotation):
        homogeneous_mat = np.zeros((4,4));
        homogeneous_mat[3,3] = 1
        homogeneous_mat[:3,3] = np.asarray(translation)
        homogeneous_mat[:3,:3] = rotation
        return HomogeneousMatrix(homogeneous_mat)

    @property
    def mat(self):
        return self._mat

    @property
    def rotation(self):
        return self._mat[:3,:3]

    @property
    def translation(self):
        return self._mat[:3,3]

    def inv(self):
        inverse_mat = np.copy(self._mat)
        inverse_mat[:3,:3] = self.rotation.T
        inverse_mat[:3,3] = -(self.rotation.T).dot(self.translation)
        return inverse_mat

class Camera(object):
    def __init__(self, extrinsic, f=350, image_resolution=(1024, 768)):
        self._extrinsic = extrinsic # camera wrt world
        self._f = f # in pixel
        u0, v0 = np.array(image_resolution)/2
        self._image_resolution = image_resolution
        self._intrinsic = np.array([[f, 0, u0, 0], [0, f, v0, 0], [0,0,1,0]])

    @property
    def intrinsic(self):
        return self._intrinsic

    def get_extrinsic(self):
        return self._extrinsic

    def project(self, key_points, ax, color):
        key_points_wrt_camera = self._extrinsic.inv().dot(key_points)
        points_in_image_frame = normalize_homogeneous_coordinates(self._intrinsic.dot(key_points_wrt_camera))
        #print(points_in_image_frame)
        image = np.zeros(self._image_resolution)

        ax.set_xlim(0, self._image_resolution[0])
        ax.set_ylim(0, self._image_resolution[1])
        ax.plot(points_in_image_frame[0,:], points_in_image_frame[1,:], 'o', color=color)
        return key_points_wrt_camera[:3,:], points_in_image_frame

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

        #print(self._camera1.get_extrinsic().mat)
        #print(self._camera1.get_extrinsic().inv())
        #print(self._camera2.get_extrinsic().mat)
        tf_cam2_wrt_cam1 = self._camera1.get_extrinsic().inv().dot(self._camera2.get_extrinsic().mat) # the translation from it is not very correct
        #tf_cam1_wrt_cam2 = self._camera2.get_extrinsic().inv().dot(self._camera1.get_extrinsic().mat)

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

