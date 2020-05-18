import numpy


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.rot2.Rot2'>.
    """

    @staticmethod
    def identity():
        # Input arrays

        # Output array
        res = [0.] * 2

        # Intermediate terms (0)

        # Output terms (2)
        res[0] = 1
        res[1] = 0

        return res

    @staticmethod
    def inverse(a):
        # Input arrays
        _a = a.data

        # Output array
        res = [0.] * 2

        # Intermediate terms (1)
        _tmp0 = 1/(_a[0]**2 + _a[1]**2)

        # Output terms (2)
        res[0] = _a[0]*_tmp0
        res[1] = -_a[1]*_tmp0

        return res

    @staticmethod
    def compose(a, b):
        # Input arrays
        _a = a.data
        _b = b.data

        # Output array
        res = [0.] * 2

        # Intermediate terms (0)

        # Output terms (2)
        res[0] = _a[0]*_b[0] - _a[1]*_b[1]
        res[1] = _a[0]*_b[1] + _a[1]*_b[0]

        return res

    @staticmethod
    def between(a, b):
        # Input arrays
        _a = a.data
        _b = b.data

        # Output array
        res = [0.] * 2

        # Intermediate terms (3)
        _tmp0 = 1/(_a[0]**2 + _a[1]**2)
        _tmp1 = _b[1]*_tmp0
        _tmp2 = _b[0]*_tmp0

        # Output terms (2)
        res[0] = _a[0]*_tmp2 + _a[1]*_tmp1
        res[1] = _a[0]*_tmp1 - _a[1]*_tmp2

        return res

