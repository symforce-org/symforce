import numpy


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geometry.rot3.Rot3'>.
    """

    @staticmethod
    def identity():
        # Input arrays

        # Output array
        res = [0.] * 4

        # Intermediate terms (0)

        # Output terms (4)
        res[0] = 0
        res[1] = 0
        res[2] = 0
        res[3] = 1

        return res

    @staticmethod
    def inverse(a):
        # Input arrays
        _a = a.storage

        # Output array
        res = [0.] * 4

        # Intermediate terms (0)

        # Output terms (4)
        res[0] = -_a[0]
        res[1] = -_a[1]
        res[2] = -_a[2]
        res[3] = _a[3]

        return res

    @staticmethod
    def compose(a, b):
        # Input arrays
        _a = a.storage
        _b = b.storage

        # Output array
        res = [0.] * 4

        # Intermediate terms (0)

        # Output terms (4)
        res[0] = _a[0]*_b[3] + _a[1]*_b[2] - _a[2]*_b[1] + _a[3]*_b[0]
        res[1] = -_a[0]*_b[2] + _a[1]*_b[3] + _a[2]*_b[0] + _a[3]*_b[1]
        res[2] = _a[0]*_b[1] - _a[1]*_b[0] + _a[2]*_b[3] + _a[3]*_b[2]
        res[3] = -_a[0]*_b[0] - _a[1]*_b[1] - _a[2]*_b[2] + _a[3]*_b[3]

        return res

    @staticmethod
    def between(a, b):
        # Input arrays
        _a = a.storage
        _b = b.storage

        # Output array
        res = [0.] * 4

        # Intermediate terms (0)

        # Output terms (4)
        res[0] = -_a[0]*_b[3] - _a[1]*_b[2] + _a[2]*_b[1] + _a[3]*_b[0]
        res[1] = _a[0]*_b[2] - _a[1]*_b[3] - _a[2]*_b[0] + _a[3]*_b[1]
        res[2] = -_a[0]*_b[1] + _a[1]*_b[0] - _a[2]*_b[3] + _a[3]*_b[2]
        res[3] = _a[0]*_b[0] + _a[1]*_b[1] + _a[2]*_b[2] + _a[3]*_b[3]

        return res

