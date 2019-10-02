import numpy as np
from math import cos, sin

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


