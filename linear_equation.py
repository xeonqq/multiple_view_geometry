class LinearEquation(object):
    def __init__(self, coefficents):
        # coefficents are in the order a ,b, c where ax + by + c = 0
        self._coefficents = coefficents

    def solve_y(self, x):
        a, b, c = self._coefficents
        assert b != 0
        return (-c-x*a)/b


