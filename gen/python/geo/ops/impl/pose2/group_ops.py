import numpy


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.pose2.Pose2'>.
    """

    @staticmethod
    def identity():
        # Input arrays

        # Output array
        res = [0.] * 4

        # Intermediate terms (0)

        # Output terms (4)
        res[0] = 1
        res[1] = 0
        res[2] = 0
        res[3] = 0

        return res

    @staticmethod
    def inverse(a):
        # Input arrays
        _a = a.data

        # Output array
        res = [0.] * 4

        # Intermediate terms (3)
        _tmp0 = (_a[0]**2 + _a[1]**2)**(-1.0)
        _tmp1 = _a[0]*_tmp0
        _tmp2 = _a[1]*_tmp0

        # Output terms (4)
        res[0] = _tmp1
        res[1] = -_tmp2
        res[2] = -_a[2]*_tmp1 - _a[3]*_tmp2
        res[3] = _a[2]*_tmp2 - _a[3]*_tmp1

        return res

    @staticmethod
    def compose(a, b):
        # Input arrays
        _a = a.data
        _b = b.data

        # Output array
        res = [0.] * 4

        # Intermediate terms (0)

        # Output terms (4)
        res[0] = _a[0]*_b[0] - _a[1]*_b[1]
        res[1] = _a[0]*_b[1] + _a[1]*_b[0]
        res[2] = _a[0]*_b[2] - _a[1]*_b[3] + _a[2]
        res[3] = _a[0]*_b[3] + _a[1]*_b[2] + _a[3]

        return res

    @staticmethod
    def between(a, b):
        # Input arrays
        _a = a.data
        _b = b.data

        # Output array
        res = [0.] * 4

        # Intermediate terms (7)
        _tmp0 = (_a[0]**2 + _a[1]**2)**(-1.0)
        _tmp1 = _b[1]*_tmp0
        _tmp2 = _b[0]*_tmp0
        _tmp3 = _a[3]*_tmp0
        _tmp4 = _a[2]*_tmp0
        _tmp5 = _b[3]*_tmp0
        _tmp6 = _b[2]*_tmp0

        # Output terms (4)
        res[0] = _a[0]*_tmp2 + _a[1]*_tmp1
        res[1] = _a[0]*_tmp1 - _a[1]*_tmp2
        res[2] = -_a[0]*_tmp4 + _a[0]*_tmp6 - _a[1]*_tmp3 + _a[1]*_tmp5
        res[3] = -_a[0]*_tmp3 + _a[0]*_tmp5 + _a[1]*_tmp4 - _a[1]*_tmp6

        return res

