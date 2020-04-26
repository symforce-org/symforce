import numpy


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geo.pose2.Pose2'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):
        # Input arrays

        # Output array
        res = [0.] * 4

        # Intermediate terms (5)
        _tmp0 = numpy.cos(vec[2])
        _tmp1 = numpy.sin(vec[2])
        _tmp2 = (epsilon + vec[2])**(-1.0)
        _tmp3 = _tmp2*(1 - _tmp0)
        _tmp4 = _tmp1*_tmp2

        # Output terms (4)
        res[0] = _tmp0
        res[1] = _tmp1
        res[2] = -_tmp3*vec[1] + _tmp4*vec[0]
        res[3] = _tmp3*vec[0] + _tmp4*vec[1]

        return res

    @staticmethod
    def to_tangent(a, epsilon):
        # Input arrays
        _a = a.data

        # Output array
        res = [0.] * 3

        # Intermediate terms (5)
        _tmp0 = numpy.arctan2(_a[1], _a[0] + epsilon*(numpy.sign(_a[0]) + 0.5))
        _tmp1 = 0.5*_tmp0
        _tmp2 = _a[3]*_tmp1
        _tmp3 = _a[2]*_tmp1
        _tmp4 = _a[1]/numpy.amax((epsilon,1 - _a[0]))

        # Output terms (3)
        res[0] = _tmp2 + _tmp3*_tmp4
        res[1] = _tmp2*_tmp4 - _tmp3
        res[2] = _tmp0

        return res

    @staticmethod
    def retract(a, vec, epsilon):
        # Input arrays
        _a = a.data

        # Output array
        res = [0.] * 4

        # Intermediate terms (8)
        _tmp0 = numpy.sin(vec[2])
        _tmp1 = numpy.cos(vec[2])
        _tmp2 = 1 - _tmp1
        _tmp3 = (epsilon + vec[2])**(-1.0)
        _tmp4 = _tmp3*vec[1]
        _tmp5 = _tmp3*vec[0]
        _tmp6 = _tmp0*_tmp5 - _tmp2*_tmp4
        _tmp7 = _tmp0*_tmp4 + _tmp2*_tmp5

        # Output terms (4)
        res[0] = _a[0]*_tmp1 - _a[1]*_tmp0
        res[1] = _a[0]*_tmp0 + _a[1]*_tmp1
        res[2] = _a[0]*_tmp6 - _a[1]*_tmp7 + _a[2]
        res[3] = _a[0]*_tmp7 + _a[1]*_tmp6 + _a[3]

        return res

    @staticmethod
    def local_coordinates(a, b, epsilon):
        # Input arrays
        _a = a.data
        _b = b.data

        # Output array
        res = [0.] * 3

        # Intermediate terms (10)
        _tmp0 = (_a[0]**2 + _a[1]**2)**(-1.0)
        _tmp1 = _a[0]*_tmp0
        _tmp2 = _a[1]*_tmp0
        _tmp3 = -_b[0]*_tmp2 + _b[1]*_tmp1
        _tmp4 = _b[0]*_tmp1 + _b[1]*_tmp2
        _tmp5 = numpy.arctan2(_tmp3, _tmp4 + epsilon*(numpy.sign(_tmp4) + 0.5))
        _tmp6 = 0.5*_tmp5
        _tmp7 = _tmp6*(_a[2]*_tmp2 - _a[3]*_tmp1 - _b[2]*_tmp2 + _b[3]*_tmp1)
        _tmp8 = _tmp6*(-_a[2]*_tmp1 - _a[3]*_tmp2 + _b[2]*_tmp1 + _b[3]*_tmp2)
        _tmp9 = _tmp3/numpy.amax((epsilon,1 - _tmp4))

        # Output terms (3)
        res[0] = _tmp7 + _tmp8*_tmp9
        res[1] = _tmp7*_tmp9 - _tmp8
        res[2] = _tmp5

        return res

