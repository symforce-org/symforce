import numpy


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geo.rot2.Rot2'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):

        # Input arrays

        # Intermediate terms

        # Output terms
        _res = [0.] * 2
        _res[0] = numpy.cos(vec[0])
        _res[1] = numpy.sin(vec[0])
        return _res

    @staticmethod
    def to_tangent(a, epsilon):

        # Input arrays
        _a = a.data

        # Intermediate terms

        # Output terms
        _res = [0.] * 1
        _res[0] = numpy.arctan2(_a[1], _a[0])
        return _res

    @staticmethod
    def retract(a, vec, epsilon):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = numpy.sin(vec[0])
        _tmp1 = numpy.cos(vec[0])

        # Output terms
        _res = [0.] * 2
        _res[0] = _a[0]*_tmp1 - _a[1]*_tmp0
        _res[1] = _a[0]*_tmp0 + _a[1]*_tmp1
        return _res

    @staticmethod
    def local_coordinates(a, b, epsilon):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = 1/(_a[0]**2 + _a[1]**2)
        _tmp1 = _a[0]*_tmp0
        _tmp2 = _a[1]*_tmp0

        # Output terms
        _res = [0.] * 1
        _res[0] = numpy.arctan2(-_b[0]*_tmp2 + _b[1]*_tmp1, _b[0]*_tmp1 + _b[1]*_tmp2)
        return _res

