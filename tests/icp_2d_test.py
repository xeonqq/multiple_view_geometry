import unittest
import numpy as  np
import matplotlib.pyplot as plt
from .context import multiple_view_geometry

from multiple_view_geometry.icp import icp_known_correspondence


def create_ground_truth(num_of_points):
    x = np.arange(0,num_of_points)
    y = np.sin(x*2*np.pi/num_of_points)*2+4
    return np.column_stack((x, y))


def distort_points(points, R, t):
    return (R.dot(points.T)+t[:, np.newaxis]).T

def add_gaussian_noise(points, mean, variance):
    noise = np.random.normal(mean, variance, points.shape)
    return points + noise
    

class ICP2dTest(unittest.TestCase):
    def setUp(self):
        self._points_gt = create_ground_truth(100);
        self._angle = np.pi/12;
        self._R  = np.array([[np.cos(self._angle), -np.sin(self._angle)],[np.sin(self._angle), np.cos(self._angle)]])
        self._t = np.array([0.02,0.01])
        self._points_obs = add_gaussian_noise(distort_points(self._points_gt, self._R, self._t), 0.01,0.02)

    def test_icp_known_correspondence(self):
        plt.plot(self._points_gt[:,0], self._points_gt[:,1], label="gt")
        plt.plot(self._points_obs[:,0], self._points_obs[:,1], label="observation")
        
        R, t = icp_known_correspondence(self._points_gt, self._points_obs)
        corrected_points = (R.dot(self._points_obs.T) + t[:,np.newaxis]).T

        plt.plot(corrected_points[:,0], corrected_points[:,1], label="correction")

        plt.legend()
        plt.show()
        np.testing.assert_array_almost_equal(self._R, R.T, decimal=2)
        np.testing.assert_array_almost_equal(self._t, -R.T.dot(t), decimal=2)


if __name__ == '__main__':
    unittest.main()
