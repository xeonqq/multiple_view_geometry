import numpy as np
from .transform_utils import points_to_homogeneous_coordinates


def icp_known_correspondence(matched_points, points_obs):
    obs = points_to_homogeneous_coordinates(points_obs.T).T
    matched = points_to_homogeneous_coordinates(matched_points.T).T
    transform = np.linalg.inv(obs.T.dot(obs)).dot(obs.T.dot(matched))
    return transform[:2,:2].T, transform[2,0:2]
    



