import numpy


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geo.pose2.Pose2'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):

        # Input arrays

        # Intermediate terms
        _tmp0 = numpy.cos(vec[2])
        _tmp1 = numpy.sin(vec[2])
        _tmp2 = (epsilon * (2 * numpy.amin((0, numpy.sign(vec[2])), axis=0) + 1) + vec[2]) ** (-1.0)
        _tmp3 = _tmp2 * (1 - _tmp0)
        _tmp4 = _tmp2 * (_tmp1 + epsilon * (2 * numpy.amin((0, numpy.sign(_tmp1)), axis=0) + 1))

        # Output terms
        _res = [0.0] * 4
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res[2] = -_tmp3 * vec[1] + _tmp4 * vec[0]
        _res[3] = _tmp3 * vec[0] + _tmp4 * vec[1]
        return _res

    @staticmethod
    def to_tangent(a, epsilon):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = numpy.arctan2(_a[1], _a[0])
        _tmp1 = 0.5 * _tmp0 + 0.5 * epsilon * (2 * numpy.amin((0, numpy.sign(_tmp0)), axis=0) + 1)
        _tmp2 = _a[3] * _tmp1
        _tmp3 = _a[2] * _tmp1
        _tmp4 = (_a[0] + 1) / (
            _a[1] + epsilon * (2 * numpy.amin((0, numpy.sign(_a[1])), axis=0) + 1)
        )

        # Output terms
        _res = [0.0] * 3
        _res[0] = _tmp2 + _tmp3 * _tmp4
        _res[1] = _tmp2 * _tmp4 - _tmp3
        _res[2] = _tmp0
        return _res

    @staticmethod
    def retract(a, vec, epsilon):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = numpy.sin(vec[2])
        _tmp1 = numpy.cos(vec[2])
        _tmp2 = (epsilon * (2 * numpy.amin((0, numpy.sign(vec[2])), axis=0) + 1) + vec[2]) ** (-1.0)
        _tmp3 = _tmp2 * (_tmp0 + epsilon * (2 * numpy.amin((0, numpy.sign(_tmp0)), axis=0) + 1))
        _tmp4 = _tmp2 * (1 - _tmp1)
        _tmp5 = _tmp3 * vec[1] + _tmp4 * vec[0]
        _tmp6 = _tmp3 * vec[0] - _tmp4 * vec[1]

        # Output terms
        _res = [0.0] * 4
        _res[0] = _a[0] * _tmp1 - _a[1] * _tmp0
        _res[1] = _a[0] * _tmp0 + _a[1] * _tmp1
        _res[2] = _a[0] * _tmp6 - _a[1] * _tmp5 + _a[2]
        _res[3] = _a[0] * _tmp5 + _a[1] * _tmp6 + _a[3]
        return _res

    @staticmethod
    def local_coordinates(a, b, epsilon):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = (_a[0] ** 2 + _a[1] ** 2) ** (-1.0)
        _tmp1 = _a[0] * _tmp0
        _tmp2 = _a[1] * _tmp0
        _tmp3 = -_b[0] * _tmp2 + _b[1] * _tmp1
        _tmp4 = _b[0] * _tmp1 + _b[1] * _tmp2
        _tmp5 = numpy.arctan2(_tmp3, _tmp4)
        _tmp6 = 0.5 * _tmp5 + 0.5 * epsilon * (2 * numpy.amin((0, numpy.sign(_tmp5)), axis=0) + 1)
        _tmp7 = _tmp6 * (_a[2] * _tmp2 - _a[3] * _tmp1 - _b[2] * _tmp2 + _b[3] * _tmp1)
        _tmp8 = _tmp6 * (-_a[2] * _tmp1 - _a[3] * _tmp2 + _b[2] * _tmp1 + _b[3] * _tmp2)
        _tmp9 = (_tmp4 + 1) / (
            _tmp3 + epsilon * (2 * numpy.amin((0, numpy.sign(_tmp3)), axis=0) + 1)
        )

        # Output terms
        _res = [0.0] * 3
        _res[0] = _tmp7 + _tmp8 * _tmp9
        _res[1] = _tmp7 * _tmp9 - _tmp8
        _res[2] = _tmp5
        return _res
