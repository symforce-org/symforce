import numpy


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geo.rot3.Rot3'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):

        # Input arrays

        # Intermediate terms
        _tmp0 = numpy.sqrt(epsilon ** 2 + vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        _tmp1 = (1.0 / 2.0) * _tmp0
        _tmp2 = numpy.sin(_tmp1) / _tmp0

        # Output terms
        _res = [0.0] * 4
        _res[0] = _tmp2 * vec[0]
        _res[1] = _tmp2 * vec[1]
        _res[2] = _tmp2 * vec[2]
        _res[3] = numpy.cos(_tmp1)
        return _res

    @staticmethod
    def to_tangent(a, epsilon):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = numpy.amin((abs(_a[3]), 1 - epsilon), axis=0)
        _tmp1 = (
            2
            * (2 * numpy.amin((0, numpy.sign(_a[3])), axis=0) + 1)
            * numpy.arccos(_tmp0)
            / numpy.sqrt(1 - _tmp0 ** 2)
        )

        # Output terms
        _res = [0.0] * 3
        _res[0] = _a[0] * _tmp1
        _res[1] = _a[1] * _tmp1
        _res[2] = _a[2] * _tmp1
        return _res

    @staticmethod
    def retract(a, vec, epsilon):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = numpy.sqrt(epsilon ** 2 + vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        _tmp1 = (1.0 / 2.0) * _tmp0
        _tmp2 = numpy.sin(_tmp1) / _tmp0
        _tmp3 = _a[1] * _tmp2
        _tmp4 = _a[2] * _tmp2
        _tmp5 = numpy.cos(_tmp1)
        _tmp6 = _a[3] * _tmp2
        _tmp7 = _a[0] * _tmp2

        # Output terms
        _res = [0.0] * 4
        _res[0] = _a[0] * _tmp5 + _tmp3 * vec[2] - _tmp4 * vec[1] + _tmp6 * vec[0]
        _res[1] = _a[1] * _tmp5 + _tmp4 * vec[0] + _tmp6 * vec[1] - _tmp7 * vec[2]
        _res[2] = _a[2] * _tmp5 - _tmp3 * vec[0] + _tmp6 * vec[2] + _tmp7 * vec[1]
        _res[3] = _a[3] * _tmp5 - _tmp3 * vec[1] - _tmp4 * vec[2] - _tmp7 * vec[0]
        return _res

    @staticmethod
    def local_coordinates(a, b, epsilon):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3]
        _tmp1 = numpy.amin((abs(_tmp0), 1 - epsilon), axis=0)
        _tmp2 = (
            2
            * (2 * numpy.amin((0, numpy.sign(_tmp0)), axis=0) + 1)
            * numpy.arccos(_tmp1)
            / numpy.sqrt(1 - _tmp1 ** 2)
        )

        # Output terms
        _res = [0.0] * 3
        _res[0] = _tmp2 * (-_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0])
        _res[1] = _tmp2 * (_a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1])
        _res[2] = _tmp2 * (-_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2])
        return _res
