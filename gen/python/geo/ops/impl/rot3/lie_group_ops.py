import numpy


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geo.rot3.Rot3'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):
        # Input arrays

        # Output array
        res = [0.] * 4

        # Intermediate terms (3)
        _tmp0 = numpy.sqrt(epsilon**2 + vec[0]**2 + vec[1]**2 + vec[2]**2)
        _tmp1 = (1./2.)*_tmp0
        _tmp2 = numpy.sin(_tmp1)/_tmp0

        # Output terms (4)
        res[0] = _tmp2*vec[0]
        res[1] = _tmp2*vec[1]
        res[2] = _tmp2*vec[2]
        res[3] = numpy.cos(_tmp1)

        return res

    @staticmethod
    def to_tangent(a, epsilon):
        # Input arrays
        _a = a.data

        # Output array
        res = [0.] * 3

        # Intermediate terms (1)
        _tmp0 = 2*numpy.arccos(numpy.amax((-1,numpy.amin((1,_a[3])))))/numpy.sqrt(numpy.amax((epsilon,-_a[3]**2 + 1)))

        # Output terms (3)
        res[0] = _a[0]*_tmp0
        res[1] = _a[1]*_tmp0
        res[2] = _a[2]*_tmp0

        return res

    @staticmethod
    def retract(a, vec, epsilon):
        # Input arrays
        _a = a.data

        # Output array
        res = [0.] * 4

        # Intermediate terms (8)
        _tmp0 = numpy.sqrt(epsilon**2 + vec[0]**2 + vec[1]**2 + vec[2]**2)
        _tmp1 = (1./2.)*_tmp0
        _tmp2 = numpy.sin(_tmp1)/_tmp0
        _tmp3 = _a[1]*_tmp2
        _tmp4 = _a[2]*_tmp2
        _tmp5 = numpy.cos(_tmp1)
        _tmp6 = _a[3]*_tmp2
        _tmp7 = _a[0]*_tmp2

        # Output terms (4)
        res[0] = _a[0]*_tmp5 + _tmp3*vec[2] - _tmp4*vec[1] + _tmp6*vec[0]
        res[1] = _a[1]*_tmp5 + _tmp4*vec[0] + _tmp6*vec[1] - _tmp7*vec[2]
        res[2] = _a[2]*_tmp5 - _tmp3*vec[0] + _tmp6*vec[2] + _tmp7*vec[1]
        res[3] = _a[3]*_tmp5 - _tmp3*vec[1] - _tmp4*vec[2] - _tmp7*vec[0]

        return res

    @staticmethod
    def local_coordinates(a, b, epsilon):
        # Input arrays
        _a = a.data
        _b = b.data

        # Output array
        res = [0.] * 3

        # Intermediate terms (2)
        _tmp0 = _a[0]*_b[0] + _a[1]*_b[1] + _a[2]*_b[2] + _a[3]*_b[3]
        _tmp1 = 2*numpy.arccos(numpy.amax((-1,numpy.amin((1,_tmp0)))))/numpy.sqrt(numpy.amax((epsilon,-_tmp0**2 + 1)))

        # Output terms (3)
        res[0] = _tmp1*(-_a[0]*_b[3] - _a[1]*_b[2] + _a[2]*_b[1] + _a[3]*_b[0])
        res[1] = _tmp1*(_a[0]*_b[2] - _a[1]*_b[3] - _a[2]*_b[0] + _a[3]*_b[1])
        res[2] = _tmp1*(-_a[0]*_b[1] + _a[1]*_b[0] - _a[2]*_b[3] + _a[3]*_b[2])

        return res

