import numpy as np

from .linear_equation import LinearEquation
from .transform_utils import normalize_homogeneous_coordinates, translation_to_skew_symetric_mat, translation_to_skew_symetric_mat, points_to_homogeneous_coordinates, create_rotation_mat_from_rpy, skew_symetric_mat_to_translation

def calculate_epipolar_line_on_other_image(point_in_camera_frame, essential_matrix, other_camera):
    epipolar_line_in_camera0 = essential_matrix.dot(point_in_camera_frame)
    lineq = LinearEquation(epipolar_line_in_camera0)
    line_end_points = []
    xs = [-2, 2]
    for x in xs:
        line_end_point = np.array([x, lineq.solve_y(x), 1])
        line_end_point = normalize_homogeneous_coordinates(other_camera.intrinsic.dot(line_end_point))
        line_end_points.append(line_end_point)
    return np.asarray(line_end_points)

def calculate_essential_matrix(camera1, camera0):
    # it calculates the essential matrix of camera1 wrt to camera0

    tf_cam1_wrt_cam0 = camera1.get_transform_wrt(camera0)

    # because of epipolar constraint
    # p0.T*E*p1 = 0, refer to https://www.youtube.com/watch?v=9fvopDHdrFg for derivation
    # Essential matrix = T_cam1_wrt_cam0 cross multiply R_cam1_wrt_cam0
    # E = t x R
    essential_matrix = translation_to_skew_symetric_mat(tf_cam1_wrt_cam0.translation).dot(tf_cam1_wrt_cam0.rotation)
    return essential_matrix

def calculate_direction_vecs_in_world_frame(points_in_image_frame, camera):
    homogeneous_points_in_camera_frame = camera.points_2d_to_homogeneous_coordinates(points_in_image_frame)
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

def solve_essential_matrix(points_in_image_frame0, camera0, points_in_image_frame1, camera1):
    # we are not going to use the extrinsic matrix from the cameras here, but we need to deduce them from the points in image frame
    # A*Vec(E) = 0
    # A is a matrix stacked with Kronecker product of (p0, p1)
    homogeneous_points_in_camera_frame0 = camera0.points_2d_to_homogeneous_coordinates(points_in_image_frame0)
    homogeneous_points_in_camera_frame1 = camera1.points_2d_to_homogeneous_coordinates(points_in_image_frame1)
    A = np.array([np.kron(point1, point0) for point0, point1 in zip(homogeneous_points_in_camera_frame0.T, homogeneous_points_in_camera_frame1.T)])
    assert np.linalg.matrix_rank(A) >= 8, "actual rank: {}".format(np.linalg.matrix_rank(A))

    _, s, vh = np.linalg.svd(A)
    E_vec= (vh.T)[:,s.size-1]
    E_est = E_vec.reshape((3,3)).T
    u, _, vh = np.linalg.svd(E_est)
    e_diagnal = np.array([1,1,0])
    s = np.diag(e_diagnal)
    projected_E_on_essential_space = u.dot(s).dot(vh)
    return projected_E_on_essential_space, u, e_diagnal, vh

def reconstruct_translation_and_rotation_from_essential_matrix(essential_matix):
    u, s, vh = np.linalg.svd(essential_matix)
    return reconstruct_translation_and_rotation_from_svd_of_essential_matrix(u, s, vh)

def reconstruct_translation_and_rotation_from_svd_of_essential_matrix(u, s, vh):
    Rz = create_rotation_mat_from_rpy(0,0,np.pi/2.0)
    Rz_neg = create_rotation_mat_from_rpy(0,0,-np.pi/2.0)
    sut = np.diag(s).dot(u.T)
    T1 = u.dot(Rz).dot(sut)
    T2 = u.dot(Rz_neg).dot(sut)
    R1 = u.dot(Rz.T).dot(vh)
    R2 = u.dot(Rz_neg.T).dot(vh)
    return skew_symetric_mat_to_translation(T1), R1, skew_symetric_mat_to_translation(T2), R2

def structure_from_motion_options(homo_points_in_image0, homo_points_in_image1, transform_cam1_wrt_cam0_option1, transform_cam1_wrt_cam0_option2):
    depth_scale_in_camera1 = structure_from_motion(homo_points_in_image0, homo_points_in_image1, transform_cam1_wrt_cam0_option1)
    correct_transform_option = transform_cam1_wrt_cam0_option1
    if depth_scale_in_camera1 is None or not (depth_scale_in_camera1 > 0).all():
        correct_transform_option = transform_cam1_wrt_cam0_option2
        depth_scale_in_camera1 = structure_from_motion(homo_points_in_image0, homo_points_in_image1, transform_cam1_wrt_cam0_option2)
    return correct_transform_option, depth_scale_in_camera1

def structure_from_motion(homo_points_in_image0, homo_points_in_image1, transform_cam1_wrt_cam0):
    """
    Returns: the 3d points in camera1's frame and the scale, assuming the translation scale between the two cameras is 1
    """
    T = transform_cam1_wrt_cam0.translation
    R = transform_cam1_wrt_cam0.rotation
    M = np.zeros((3*len(homo_points_in_image1.T), len(homo_points_in_image1.T) + 1))
    for i, (homo_point_in_image0, homo_point_in_image1) in enumerate(zip(homo_points_in_image0.T, homo_points_in_image1.T)):
        skew_symetrix_homo_point_in_image0 = translation_to_skew_symetric_mat(homo_point_in_image0)
        M[3*i: 3*(i+1), i] = skew_symetrix_homo_point_in_image0.dot(R).dot(homo_point_in_image1)
        M[3*i: 3*(i+1), -1] = skew_symetrix_homo_point_in_image0.dot(T)

    # We need to solve M*Lambda = 0
    # M (3*n, n+1), Lambda (n+1, 1)
    # We try to solve least square error problem, minimize |M*Lambda|^2, the solution is the last column of V from svd(M)
    u, s, vh = np.linalg.svd(M)
    Lambda = vh.T[:, -1]
    has_solution = not np.isclose(Lambda[-1], 0)
    if has_solution:
        # assume scale on camera translation is 1
        depth_scale_in_camera1 = Lambda[:-1] / Lambda[-1]
        return depth_scale_in_camera1
    else:
        return None
