import numpy as np
from transform_utils import normalize_homogeneous_coordinates

class Camera(object):
    def __init__(self, name, extrinsic, f=350, image_resolution=(1024, 768)):
        self._extrinsic = extrinsic # camera wrt world
        self._f = f # in pixel
        u0, v0 = np.array(image_resolution)/2
        self._image_resolution = image_resolution
        self._intrinsic = np.array([[f, 0, u0, 0], [0, f, v0, 0], [0,0,1,0]])
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def image_resolution(self):
        return self._image_resolution

    @property
    def intrinsic(self):
        return self._intrinsic

    @property
    def extrinsic(self):
        return self._extrinsic

    def project(self, key_points):
        key_points_wrt_camera = self._extrinsic.inv().dot(key_points)
        points_in_image_frame = normalize_homogeneous_coordinates(self._intrinsic.dot(key_points_wrt_camera))
        #print(points_in_image_frame)
        image = np.zeros(self._image_resolution)

        return key_points_wrt_camera[:3,:], points_in_image_frame


