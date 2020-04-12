import numpy


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geometry.rot2.Rot2'>.
    """

    @staticmethod
    def expmap(vec, epsilon):
        # Input arrays

        # Output array
        res = [0.] * 2

        # Intermediate terms (0)

        # Output terms (2)
        res[0] = numpy.cos(vec[0])
        res[1] = numpy.sin(vec[0])

        return res

    @staticmethod
    def logmap(a, epsilon):
        # Input arrays
        _a = a.storage

        # Output array
        res = [0.] * 1

        # Intermediate terms (0)

        # Output terms (1)
        res[0] = numpy.arctan2(_a[1], _a[0] + epsilon)

        return res

    @staticmethod
    def retract(a, vec, epsilon):
        # Input arrays
        _a = a.storage

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
        _a = a.storage
        _b = b.storage

        # Output array
        res = [0.] * 1

        # Intermediate terms (3)
        _tmp0 = (_a[0]**2 + _a[1]**2)**(-1.0)
        _tmp1 = _b[1]*_tmp0
        _tmp2 = _b[0]*_tmp0

        # Output terms (1)
        res[0] = numpy.arctan2(_a[0]*_tmp1 - _a[1]*_tmp2, _a[0]*_tmp2 + _a[1]*_tmp1 + epsilon)

        return res

