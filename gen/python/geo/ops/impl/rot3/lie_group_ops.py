import numpy


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geometry.rot3.Rot3'>.
    """

    @staticmethod
    def expmap(vec, epsilon):
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
    def logmap(a, epsilon):
        # Input arrays
        _a = a.storage

        # Output array
        res = [0.] * 3

        # Intermediate terms (2)
        _tmp0 = numpy.sqrt(_a[0]**2 + _a[1]**2 + _a[2]**2 + epsilon)
        _tmp1 = 2*numpy.arctan(_tmp0/(_a[3] + epsilon))/_tmp0

        # Output terms (3)
        res[0] = _a[0]*_tmp1
        res[1] = _a[1]*_tmp1
        res[2] = _a[2]*_tmp1

        return res

    @staticmethod
    def retract(a, vec, epsilon):
        # Input arrays
        _a = a.storage

        # Output array
        res = [0.] * 4

        # Intermediate terms (9)
        _tmp0 = numpy.sqrt(epsilon**2 + vec[0]**2 + vec[1]**2 + vec[2]**2)
        _tmp1 = _tmp0**(-1.0)
        _tmp2 = (1./2.)*_tmp0
        _tmp3 = numpy.sin(_tmp2)
        _tmp4 = _tmp1*_tmp3
        _tmp5 = _a[1]*_tmp4
        _tmp6 = _a[2]*_tmp4
        _tmp7 = numpy.cos(_tmp2)
        _tmp8 = _a[0]*_tmp4

        # Output terms (4)
        res[0] = _a[0]*_tmp7 + _a[3]*_tmp1*_tmp3*vec[0] + _tmp5*vec[2] - _tmp6*vec[1]
        res[1] = _a[1]*_tmp7 + _a[3]*_tmp4*vec[1] + _tmp6*vec[0] - _tmp8*vec[2]
        res[2] = _a[2]*_tmp7 + _a[3]*_tmp4*vec[2] - _tmp5*vec[0] + _tmp8*vec[1]
        res[3] = _a[3]*_tmp7 - _tmp5*vec[1] - _tmp6*vec[2] - _tmp8*vec[0]

        return res

    @staticmethod
    def local_coordinates(a, b, epsilon):
        # Input arrays
        _a = a.storage
        _b = b.storage

        # Output array
        res = [0.] * 3

        # Intermediate terms (5)
        _tmp0 = -_a[0]*_b[3] - _a[1]*_b[2] + _a[2]*_b[1] + _a[3]*_b[0]
        _tmp1 = _a[0]*_b[2] - _a[1]*_b[3] - _a[2]*_b[0] + _a[3]*_b[1]
        _tmp2 = -_a[0]*_b[1] + _a[1]*_b[0] - _a[2]*_b[3] + _a[3]*_b[2]
        _tmp3 = numpy.sqrt(_tmp0**2 + _tmp1**2 + _tmp2**2 + epsilon)
        _tmp4 = 2*numpy.arctan(_tmp3/(_a[0]*_b[0] + _a[1]*_b[1] + _a[2]*_b[2] + _a[3]*_b[3] + epsilon))/_tmp3

        # Output terms (3)
        res[0] = _tmp0*_tmp4
        res[1] = _tmp1*_tmp4
        res[2] = _tmp2*_tmp4

        return res

