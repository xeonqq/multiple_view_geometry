import numpy as np


class HomogeneousMatrix(object):
    def __init__(self, mat):
        self._mat = mat

    @staticmethod
    def create(translation, rotation):
        homogeneous_mat = np.zeros((4, 4))
        homogeneous_mat[3, 3] = 1
        homogeneous_mat[:3, 3] = np.asarray(translation)
        homogeneous_mat[:3, :3] = rotation
        return HomogeneousMatrix(homogeneous_mat)

    @property
    def mat(self):
        return self._mat

    @property
    def rotation(self):
        return self._mat[:3, :3]

    @property
    def translation(self):
        return self._mat[:3, 3]

    def inv(self):
        inverse_mat = np.copy(self._mat)
        inverse_mat[:3, :3] = self.rotation.T
        inverse_mat[:3, 3] = -(self.rotation.T).dot(self.translation)
        return inverse_mat
