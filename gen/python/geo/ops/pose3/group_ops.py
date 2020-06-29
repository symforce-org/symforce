import numpy

class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.pose3.Pose3'>.
    """

    @staticmethod
    def identity():

        # Input arrays

        # Intermediate terms

        # Output terms
        _res = [0.] * 7
        _res[0] = 0
        _res[1] = 0
        _res[2] = 0
        _res[3] = 1
        _res[4] = 0
        _res[5] = 0
        _res[6] = 0
        return _res

    @staticmethod
    def inverse(a):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = -2*_a[1]**2
        _tmp1 = -2*_a[2]**2
        _tmp2 = 2*_a[1]
        _tmp3 = _a[0]*_tmp2
        _tmp4 = 2*_a[3]
        _tmp5 = _a[2]*_tmp4
        _tmp6 = 2*_a[0]*_a[2]
        _tmp7 = _a[3]*_tmp2
        _tmp8 = -2*_a[0]**2 + 1
        _tmp9 = _a[2]*_tmp2
        _tmp10 = _a[0]*_tmp4

        # Output terms
        _res = [0.] * 7
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        _res[4] = -_a[4]*(_tmp0 + _tmp1 + 1) - _a[5]*(_tmp3 + _tmp5) - _a[6]*(_tmp6 - _tmp7)
        _res[5] = -_a[4]*(_tmp3 - _tmp5) - _a[5]*(_tmp1 + _tmp8) - _a[6]*(_tmp10 + _tmp9)
        _res[6] = -_a[4]*(_tmp6 + _tmp7) - _a[5]*(-_tmp10 + _tmp9) - _a[6]*(_tmp0 + _tmp8)
        return _res

    @staticmethod
    def compose(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = -2*_a[2]**2
        _tmp1 = -2*_a[1]**2 + 1
        _tmp2 = 2*_a[0]
        _tmp3 = _a[1]*_tmp2
        _tmp4 = 2*_a[2]
        _tmp5 = _a[3]*_tmp4
        _tmp6 = _a[0]*_tmp4
        _tmp7 = 2*_a[1]*_a[3]
        _tmp8 = -2*_a[0]**2
        _tmp9 = _a[1]*_tmp4
        _tmp10 = _a[3]*_tmp2

        # Output terms
        _res = [0.] * 7
        _res[0] = _a[0]*_b[3] + _a[1]*_b[2] - _a[2]*_b[1] + _a[3]*_b[0]
        _res[1] = -_a[0]*_b[2] + _a[1]*_b[3] + _a[2]*_b[0] + _a[3]*_b[1]
        _res[2] = _a[0]*_b[1] - _a[1]*_b[0] + _a[2]*_b[3] + _a[3]*_b[2]
        _res[3] = -_a[0]*_b[0] - _a[1]*_b[1] - _a[2]*_b[2] + _a[3]*_b[3]
        _res[4] = _a[4] + _b[4]*(_tmp0 + _tmp1) + _b[5]*(_tmp3 - _tmp5) + _b[6]*(_tmp6 + _tmp7)
        _res[5] = _a[5] + _b[4]*(_tmp3 + _tmp5) + _b[5]*(_tmp0 + _tmp8 + 1) + _b[6]*(-_tmp10 + _tmp9)
        _res[6] = _a[6] + _b[4]*(_tmp6 - _tmp7) + _b[5]*(_tmp10 + _tmp9) + _b[6]*(_tmp1 + _tmp8)
        return _res

    @staticmethod
    def between(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = -2*_a[1]**2
        _tmp1 = -2*_a[2]**2 + 1
        _tmp2 = _tmp0 + _tmp1
        _tmp3 = 2*_a[0]*_a[1]
        _tmp4 = 2*_a[2]
        _tmp5 = _a[3]*_tmp4
        _tmp6 = _tmp3 + _tmp5
        _tmp7 = _a[0]*_tmp4
        _tmp8 = 2*_a[3]
        _tmp9 = _a[1]*_tmp8
        _tmp10 = _tmp7 - _tmp9
        _tmp11 = _tmp3 - _tmp5
        _tmp12 = -2*_a[0]**2
        _tmp13 = _tmp1 + _tmp12
        _tmp14 = _a[1]*_tmp4
        _tmp15 = _a[0]*_tmp8
        _tmp16 = _tmp14 + _tmp15
        _tmp17 = _tmp0 + _tmp12 + 1
        _tmp18 = _tmp7 + _tmp9
        _tmp19 = _tmp14 - _tmp15

        # Output terms
        _res = [0.] * 7
        _res[0] = -_a[0]*_b[3] - _a[1]*_b[2] + _a[2]*_b[1] + _a[3]*_b[0]
        _res[1] = _a[0]*_b[2] - _a[1]*_b[3] - _a[2]*_b[0] + _a[3]*_b[1]
        _res[2] = -_a[0]*_b[1] + _a[1]*_b[0] - _a[2]*_b[3] + _a[3]*_b[2]
        _res[3] = _a[0]*_b[0] + _a[1]*_b[1] + _a[2]*_b[2] + _a[3]*_b[3]
        _res[4] = -_a[4]*_tmp2 - _a[5]*_tmp6 - _a[6]*_tmp10 + _b[4]*_tmp2 + _b[5]*_tmp6 + _b[6]*_tmp10
        _res[5] = -_a[4]*_tmp11 - _a[5]*_tmp13 - _a[6]*_tmp16 + _b[4]*_tmp11 + _b[5]*_tmp13 + _b[6]*_tmp16
        _res[6] = -_a[4]*_tmp18 - _a[5]*_tmp19 - _a[6]*_tmp17 + _b[4]*_tmp18 + _b[5]*_tmp19 + _b[6]*_tmp17
        return _res

