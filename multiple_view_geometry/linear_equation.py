class LinearEquation(object):
    def __init__(self, coefficents):
        # coefficents are in the order [a ,b, c] or of shape (3, N) where ax + by + c = 0
        self._coefficents = coefficents

    def solve_y(self, x):
        if self._coefficents.ndim == 1:
            a, b, c = self._coefficents
            assert b != 0
        else:
            a = self._coefficents[:,:2]
            b = self._coefficents[:,:2]
            c = self._coefficents[:,:2]
        return (-c-x*a)/b


