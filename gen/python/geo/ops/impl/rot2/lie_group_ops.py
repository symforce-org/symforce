import numpy


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geo.rot2.Rot2'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):
        # Input arrays

        # Output array
        res = [0.] * 2

        # Intermediate terms (0)

        # Output terms (2)
        res[0] = numpy.cos(vec[0])
        res[1] = numpy.sin(vec[0])

        return res

    @staticmethod
    def to_tangent(a, epsilon):
        # Input arrays
        _a = a.data

        # Output array
        res = [0.] * 1

        # Intermediate terms (0)

        # Output terms (1)
        res[0] = numpy.arctan2(_a[1], _a[0] + epsilon*(numpy.sign(_a[0]) + 0.5))

        return res

    @staticmethod
    def retract(a, vec, epsilon):
        # Input arrays
        _a = a.data

        # Output array
        res = [0.] * 2

        # Intermediate terms (2)
        _tmp0 = numpy.sin(vec[0])
        _tmp1 = numpy.cos(vec[0])

        # Output terms (2)
        res[0] = _a[0]*_tmp1 - _a[1]*_tmp0
        res[1] = _a[0]*_tmp0 + _a[1]*_tmp1

        return res

    @staticmethod
    def local_coordinates(a, b, epsilon):
        # Input arrays
        _a = a.data
        _b = b.data

        # Output array
        res = [0.] * 1

        # Intermediate terms (4)
        _tmp0 = (_a[0]**2 + _a[1]**2)**(-1.0)
        _tmp1 = _b[1]*_tmp0
        _tmp2 = _b[0]*_tmp0
        _tmp3 = _a[0]*_tmp2 + _a[1]*_tmp1

        # Output terms (1)
        res[0] = numpy.arctan2(_a[0]*_tmp1 - _a[1]*_tmp2, _tmp3 + epsilon*(numpy.sign(_tmp3) + 0.5))

        return res

