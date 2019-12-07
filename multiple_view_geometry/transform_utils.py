import numpy as np
from math import cos, sin, pi, atan2, asin


def create_rotation_mat_from_rpy(roll, pitch, yaw):
    Rx = np.array([1, 0, 0, 0, cos(roll), -sin(roll), 0, sin(roll), cos(roll)]).reshape((3, 3))
    Ry = np.array([cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch)]).reshape((3, 3))
    Rz = np.array([cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1]).reshape((3, 3))
    R = Rz.dot(Ry.dot(Rx))
    return R


def isclose(x, y, rtol=1.0e-5, atol=1.0e-8):
    return abs(x - y) <= atol + rtol * abs(y)


def euler_angles_from_rotation_matrix(R):
    """
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    https://www.gregslabaugh.net/publications/euler.pdf
    """
    phi = 0.0
    if isclose(R[2, 0], -1.0):
        theta = pi / 2.0
        psi = atan2(R[0, 1], R[0, 2])
    elif isclose(R[2, 0], 1.0):
        theta = -pi / 2.0
        psi = atan2(-R[0, 1], -R[0, 2])
    else:
        theta = -asin(R[2, 0])
        cos_theta = cos(theta)
        psi = atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
        phi = atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
    return psi, theta, phi  # roll, pitch, yaw


def points_to_homogeneous_coordinates(points, homogeneous_value=1):
    dim, num = points.shape
    return np.vstack((points, np.full((1, num), homogeneous_value)))


def normalize_homogeneous_coordinates(points):
    normalized_points = []
    if points.ndim == 2:
        normalized_points = points[:-1, :] / points[-1, :]
    elif points.ndim == 1:
        normalized_points = points[:-1] / points[-1]
    return normalized_points


def translation_to_skew_symetric_mat(translation):
    a1, a2, a3 = translation
    return np.array([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])


def skew_symetric_mat_to_translation(skew_symetric_mat):
    return np.array([skew_symetric_mat[2, 1], skew_symetric_mat[0, -1], skew_symetric_mat[1, 0]])
