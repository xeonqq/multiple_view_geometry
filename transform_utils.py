import numpy as np
from math import cos, sin
from linear_equation import LinearEquation

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

def points_to_homogeneous_coordinates(points, homogeneous_value=1):
    dim, num = points.shape
    return np.vstack((points, np.full((1, num), homogeneous_value)))

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

def calculate_epipolar_line_on_other_image(point_in_camera_frame, essential_matrix, other_camera):
    epipolar_line_in_camera0 = essential_matrix.dot(point_in_camera_frame)
    lineq = LinearEquation(epipolar_line_in_camera0)
    line_end_points = []
    xs = [-2, 2]
    for x in xs:
        line_end_point = np.array([x, lineq.solve_y(x), 1])
        line_end_point = normalize_homogeneous_coordinates(other_camera.intrinsic[:,:3].dot(line_end_point))
        line_end_points.append(line_end_point)
    return np.asarray(line_end_points)

def calculate_essential_matrix(camera1, camera0):
    # it calculates the essential matrix of camera1 wrt to camera0

    # tf_cam1_wrt_cam0 = tf_world_wrt_cam0 * tf_cam1_wrt_world
    tf_cam1_wrt_cam0 = camera0.extrinsic.inv().dot(camera1.extrinsic.mat)

    R_cam1_wrt_cam0 = tf_cam1_wrt_cam0[:3,:3]
    T_cam1_wrt_cam0 = tf_cam1_wrt_cam0[:3,3]

    # because of epipolar constraint
    # p0.T*E*p1 = 0, refer to https://www.youtube.com/watch?v=9fvopDHdrFg for derivation
    # Essential matrix = T_cam1_wrt_cam0 cross multiply R_cam1_wrt_cam0
    # E = t x R
    essential_matrix = translation_to_skew_symetric_mat(T_cam1_wrt_cam0).dot(R_cam1_wrt_cam0)
    return essential_matrix

def calculate_direction_vecs_in_world_frame(points_in_image_frame, camera):
    points_normalized_to_pixel_center = points_in_image_frame - camera.pixel_center[:,np.newaxis]
    homogeneous_points_in_camera_frame = points_to_homogeneous_coordinates(points_normalized_to_pixel_center, camera.focal_length_in_pixels)/camera.focal_length_in_pixels
    homogeneous_points_in_camera_frame = points_to_homogeneous_coordinates(homogeneous_points_in_camera_frame)
    homogeneous_points_in_world_frame = camera.extrinsic.mat.dot(homogeneous_points_in_camera_frame)
    points_in_world_frame = normalize_homogeneous_coordinates(homogeneous_points_in_world_frame)[:3,:]
    direction_vectors = points_in_world_frame - camera.extrinsic.translation[:, np.newaxis]
    return direction_vectors

def triangulate(start_points0, direction_vecs0, start_points1, direction_vecs1):
    # for calculating intersection of 3d lines
    # refer to http://geomalgorithms.com/a05-_intersect-1.html
    perpendicular_vecs = np.cross(direction_vecs0.T, direction_vecs1.T)
    vecs_perpendicular_to_vecs0 = np.cross(perpendicular_vecs, direction_vecs0.T).T
    w = start_points1 - start_points0
    numerator = np.sum(-vecs_perpendicular_to_vecs0 * w[:,np.newaxis],axis=0)
    denominator =  np.sum(vecs_perpendicular_to_vecs0 * direction_vecs1,axis=0)
    ss = numerator / denominator
    intersection_points = start_points1[:,np.newaxis] + direction_vecs1 * ss

    return intersection_points

def reconstruct_3d_points(points_in_image_frame0, camera0, points_in_image_frame1, camera1):
    dir_vecs0 = calculate_direction_vecs_in_world_frame( \
        points_in_image_frame0, camera0)
    dir_vecs1 = calculate_direction_vecs_in_world_frame( \
        points_in_image_frame1, camera1)
    points_3d = triangulate(camera0.extrinsic.translation, dir_vecs0, \
        camera1.extrinsic.translation, dir_vecs1)
    return points_3d


