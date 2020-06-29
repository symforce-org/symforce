import numpy

class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.rot3.Rot3'>.
    """

    @staticmethod
    def identity():

        # Input arrays

        # Intermediate terms

        # Output terms
        _res = [0.] * 4
        _res[0] = 0
        _res[1] = 0
        _res[2] = 0
        _res[3] = 1
        return _res

    @staticmethod
    def inverse(a):

        # Input arrays
        _a = a.data

        # Intermediate terms

        # Output terms
        _res = [0.] * 4
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        return _res

    @staticmethod
    def compose(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms

        # Output terms
        _res = [0.] * 4
        _res[0] = _a[0]*_b[3] + _a[1]*_b[2] - _a[2]*_b[1] + _a[3]*_b[0]
        _res[1] = -_a[0]*_b[2] + _a[1]*_b[3] + _a[2]*_b[0] + _a[3]*_b[1]
        _res[2] = _a[0]*_b[1] - _a[1]*_b[0] + _a[2]*_b[3] + _a[3]*_b[2]
        _res[3] = -_a[0]*_b[0] - _a[1]*_b[1] - _a[2]*_b[2] + _a[3]*_b[3]
        return _res

    @staticmethod
    def between(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms

        # Output terms
        _res = [0.] * 4
        _res[0] = -_a[0]*_b[3] - _a[1]*_b[2] + _a[2]*_b[1] + _a[3]*_b[0]
        _res[1] = _a[0]*_b[2] - _a[1]*_b[3] - _a[2]*_b[0] + _a[3]*_b[1]
        _res[2] = -_a[0]*_b[1] + _a[1]*_b[0] - _a[2]*_b[3] + _a[3]*_b[2]
        _res[3] = _a[0]*_b[0] + _a[1]*_b[1] + _a[2]*_b[2] + _a[3]*_b[3]
        return _res

