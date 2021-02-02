import numpy


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.rot2.Rot2'>.
    """

    @staticmethod
    def identity():

        # Input arrays

        # Intermediate terms

        # Output terms
        _res = [0.0] * 2
        _res[0] = 1
        _res[1] = 0
        return _res

    @staticmethod
    def inverse(a):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = (_a[0] ** 2 + _a[1] ** 2) ** (-1.0)

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0] * _tmp0
        _res[1] = -_a[1] * _tmp0
        return _res

    @staticmethod
    def compose(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms

        # Output terms
        _res = [0.0] * 2
        _res[0] = _a[0] * _b[0] - _a[1] * _b[1]
        _res[1] = _a[0] * _b[1] + _a[1] * _b[0]
        return _res

    @staticmethod
    def between(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = (_a[0] ** 2 + _a[1] ** 2) ** (-1.0)
        _tmp1 = _a[1] * _tmp0
        _tmp2 = _a[0] * _tmp0

        # Output terms
        _res = [0.0] * 2
        _res[0] = _b[0] * _tmp2 + _b[1] * _tmp1
        _res[1] = -_b[0] * _tmp1 + _b[1] * _tmp2
        return _res

    @staticmethod
    def inverse_with_jacobian(a):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = _a[1] ** 2
        _tmp1 = _a[0] ** 2
        _tmp2 = _tmp0 + _tmp1
        _tmp3 = _tmp2 ** (-1.0)
        _tmp4 = _a[0] * _tmp3
        _tmp5 = _a[1] * _tmp3
        _tmp6 = 2 / _tmp2 ** 2
        _tmp7 = 2 / _tmp2 ** 3

        # Output terms
        _res = [0.0] * 2
        _res[0] = _tmp4
        _res[1] = -_tmp5
        _res_D_a = [0.0] * 1
        _res_D_a[0] = _a[0] * (-_a[0] * _tmp0 * _tmp7 + _tmp4 * (_tmp0 * _tmp6 - _tmp3)) - _a[1] * (
            _a[1] * _tmp1 * _tmp7 + _tmp5 * (-_tmp1 * _tmp6 + _tmp3)
        )
        return _res, _res_D_a

    @staticmethod
    def compose_with_jacobians(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = _a[0] * _b[0] - _a[1] * _b[1]
        _tmp1 = _a[0] * _b[1] + _a[1] * _b[0]

        # Output terms
        _res = [0.0] * 2
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res_D_a = [0.0] * 1
        _res_D_a[0] = _a[0] * (_b[0] * _tmp0 + _b[1] * _tmp1) - _a[1] * (
            -_b[0] * _tmp1 + _b[1] * _tmp0
        )
        _res_D_b = [0.0] * 1
        _res_D_b[0] = _b[0] * (_a[0] * _tmp0 + _a[1] * _tmp1) - _b[1] * (
            -_a[0] * _tmp1 + _a[1] * _tmp0
        )
        return _res, _res_D_a, _res_D_b

    @staticmethod
    def between_with_jacobians(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = _a[1] ** 2
        _tmp1 = _a[0] ** 2
        _tmp2 = _tmp0 + _tmp1
        _tmp3 = _tmp2 ** (-1.0)
        _tmp4 = _b[1] * _tmp3
        _tmp5 = _b[0] * _tmp3
        _tmp6 = _a[0] * _tmp5 + _a[1] * _tmp4
        _tmp7 = _a[0] * _tmp4 - _a[1] * _tmp5
        _tmp8 = 2 / _tmp2 ** 2
        _tmp9 = _tmp0 * _tmp8
        _tmp10 = _a[0] * _a[1] * _tmp8
        _tmp11 = -_b[1] * _tmp10
        _tmp12 = _b[0] * _tmp10
        _tmp13 = _tmp1 * _tmp8
        _tmp14 = _tmp3 * _tmp6
        _tmp15 = _tmp3 * _tmp7

        # Output terms
        _res = [0.0] * 2
        _res[0] = _tmp6
        _res[1] = _tmp7
        _res_D_a = [0.0] * 1
        _res_D_a[0] = _a[0] * (
            _tmp6 * (_b[0] * _tmp9 + _tmp11 - _tmp5) - _tmp7 * (-_b[1] * _tmp9 - _tmp12 + _tmp4)
        ) - _a[1] * (
            _tmp6 * (-_b[1] * _tmp13 + _tmp12 + _tmp4) - _tmp7 * (-_b[0] * _tmp13 + _tmp11 + _tmp5)
        )
        _res_D_b = [0.0] * 1
        _res_D_b[0] = _b[0] * (_a[0] * _tmp14 - _a[1] * _tmp15) - _b[1] * (
            -_a[0] * _tmp15 - _a[1] * _tmp14
        )
        return _res, _res_D_a, _res_D_b
