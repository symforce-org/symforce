import numpy


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.pose2.Pose2'>.
    """

    @staticmethod
    def identity():

        # Input arrays

        # Intermediate terms

        # Output terms
        _res = [0.0] * 4
        _res[0] = 1
        _res[1] = 0
        _res[2] = 0
        _res[3] = 0
        return _res

    @staticmethod
    def inverse(a):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = (_a[0] ** 2 + _a[1] ** 2) ** (-1.0)
        _tmp1 = _a[0] * _tmp0
        _tmp2 = _a[1] * _tmp0

        # Output terms
        _res = [0.0] * 4
        _res[0] = _tmp1
        _res[1] = -_tmp2
        _res[2] = -_a[2] * _tmp1 - _a[3] * _tmp2
        _res[3] = _a[2] * _tmp2 - _a[3] * _tmp1
        return _res

    @staticmethod
    def compose(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms

        # Output terms
        _res = [0.0] * 4
        _res[0] = _a[0] * _b[0] - _a[1] * _b[1]
        _res[1] = _a[0] * _b[1] + _a[1] * _b[0]
        _res[2] = _a[0] * _b[2] - _a[1] * _b[3] + _a[2]
        _res[3] = _a[0] * _b[3] + _a[1] * _b[2] + _a[3]
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
        _tmp3 = _a[3] * _tmp0
        _tmp4 = _a[2] * _tmp0

        # Output terms
        _res = [0.0] * 4
        _res[0] = _b[0] * _tmp2 + _b[1] * _tmp1
        _res[1] = -_b[0] * _tmp1 + _b[1] * _tmp2
        _res[2] = -_a[0] * _tmp4 - _a[1] * _tmp3 + _b[2] * _tmp2 + _b[3] * _tmp1
        _res[3] = -_a[0] * _tmp3 + _a[1] * _tmp4 - _b[2] * _tmp1 + _b[3] * _tmp2
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
        _tmp6 = _a[3] * _tmp3
        _tmp7 = _a[2] * _tmp3
        _tmp8 = _tmp2 ** (-2.0)
        _tmp9 = _tmp0 * _tmp8
        _tmp10 = _tmp1 * _tmp8
        _tmp11 = -_tmp10 - _tmp9
        _tmp12 = _a[0] * _tmp11
        _tmp13 = _a[1] * _tmp11
        _tmp14 = 2 * _tmp9
        _tmp15 = 2 * _a[0] * _a[1] * _tmp8
        _tmp16 = -_a[3] * _tmp15
        _tmp17 = _a[2] * _tmp14 + _tmp16 - _tmp7
        _tmp18 = _a[2] * _tmp15
        _tmp19 = -_a[3] * _tmp14 - _tmp18 + _tmp6
        _tmp20 = 2 * _tmp10
        _tmp21 = -_a[3] * _tmp20 + _tmp18 + _tmp6
        _tmp22 = -_a[2] * _tmp20 + _tmp16 + _tmp7
        _tmp23 = 2 / _tmp2 ** 3

        # Output terms
        _res = [0.0] * 4
        _res[0] = _tmp4
        _res[1] = -_tmp5
        _res[2] = -_a[0] * _tmp7 - _a[1] * _tmp6
        _res[3] = -_a[0] * _tmp6 + _a[1] * _tmp7
        _res_D_a = numpy.zeros((3, 3))
        _res_D_a[0, 0] = _tmp12
        _res_D_a[0, 1] = -_tmp13
        _res_D_a[0, 2] = _a[0] * (_tmp17 * _tmp5 - _tmp19 * _tmp4) - _a[1] * (
            _tmp21 * _tmp5 - _tmp22 * _tmp4
        )
        _res_D_a[1, 0] = _tmp13
        _res_D_a[1, 1] = _tmp12
        _res_D_a[1, 2] = _a[0] * (-_tmp17 * _tmp4 - _tmp19 * _tmp5) - _a[1] * (
            -_tmp21 * _tmp4 - _tmp22 * _tmp5
        )
        _res_D_a[2, 0] = 0
        _res_D_a[2, 1] = 0
        _res_D_a[2, 2] = _a[0] * (-_a[0] * _tmp0 * _tmp23 + _tmp4 * (_tmp14 - _tmp3)) - _a[1] * (
            _a[1] * _tmp1 * _tmp23 + _tmp5 * (-_tmp20 + _tmp3)
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
        _tmp2 = _a[0] * _tmp0 + _a[1] * _tmp1
        _tmp3 = _a[0] * _tmp1
        _tmp4 = _a[1] * _tmp0
        _tmp5 = _tmp3 - _tmp4
        _tmp6 = _b[2] * _tmp1
        _tmp7 = _b[3] * _tmp0
        _tmp8 = _b[2] * _tmp0 + _b[3] * _tmp1
        _tmp9 = -_tmp3 + _tmp4
        _tmp10 = _b[0] * _tmp2
        _tmp11 = _b[1] * _tmp2
        _tmp12 = -_b[1] * _tmp9 + _tmp10

        # Output terms
        _res = [0.0] * 4
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res[2] = _a[0] * _b[2] - _a[1] * _b[3] + _a[2]
        _res[3] = _a[0] * _b[3] + _a[1] * _b[2] + _a[3]
        _res_D_a = numpy.zeros((3, 3))
        _res_D_a[0, 0] = _tmp2
        _res_D_a[0, 1] = _tmp5
        _res_D_a[0, 2] = _a[0] * (_tmp6 - _tmp7) - _a[1] * _tmp8
        _res_D_a[1, 0] = _tmp9
        _res_D_a[1, 1] = _tmp2
        _res_D_a[1, 2] = _a[0] * _tmp8 - _a[1] * (-_tmp6 + _tmp7)
        _res_D_a[2, 0] = 0
        _res_D_a[2, 1] = 0
        _res_D_a[2, 2] = _a[0] * (_b[0] * _tmp0 + _b[1] * _tmp1) - _a[1] * (
            -_b[0] * _tmp1 + _b[1] * _tmp0
        )
        _res_D_b = numpy.zeros((3, 3))
        _res_D_b[0, 0] = _b[1] * _tmp5 + _tmp10
        _res_D_b[0, 1] = _b[0] * _tmp5 - _tmp11
        _res_D_b[0, 2] = 0
        _res_D_b[1, 0] = _b[0] * _tmp9 + _tmp11
        _res_D_b[1, 1] = _tmp12
        _res_D_b[1, 2] = 0
        _res_D_b[2, 0] = 0
        _res_D_b[2, 1] = 0
        _res_D_b[2, 2] = _tmp12
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
        _tmp8 = _a[3] * _tmp3
        _tmp9 = _a[2] * _tmp3
        _tmp10 = _b[3] * _tmp3
        _tmp11 = _b[2] * _tmp3
        _tmp12 = _tmp3 * _tmp7
        _tmp13 = _a[0] * _tmp12
        _tmp14 = _tmp3 * _tmp6
        _tmp15 = _a[1] * _tmp14
        _tmp16 = -_tmp13 - _tmp15
        _tmp17 = _a[1] * _tmp12
        _tmp18 = _a[0] * _tmp14
        _tmp19 = _tmp17 - _tmp18
        _tmp20 = _a[0] * _tmp19
        _tmp21 = _a[1] * _tmp19
        _tmp22 = 2 / _tmp2 ** 2
        _tmp23 = _tmp0 * _tmp22
        _tmp24 = _a[0] * _a[1] * _tmp22
        _tmp25 = _a[3] * _tmp24 - _b[3] * _tmp24
        _tmp26 = -_a[2] * _tmp23 + _b[2] * _tmp23 - _tmp11 + _tmp25 + _tmp9
        _tmp27 = _a[2] * _tmp24
        _tmp28 = _b[2] * _tmp24
        _tmp29 = _tmp10 - _tmp8
        _tmp30 = _a[3] * _tmp23 - _b[3] * _tmp23 + _tmp27 - _tmp28 + _tmp29
        _tmp31 = _tmp1 * _tmp22
        _tmp32 = _a[3] * _tmp31 - _b[3] * _tmp31 - _tmp27 + _tmp28 + _tmp29
        _tmp33 = _a[2] * _tmp31 - _b[2] * _tmp31 + _tmp11 + _tmp25 - _tmp9
        _tmp34 = _tmp13 + _tmp15
        _tmp35 = -_b[1] * _tmp24
        _tmp36 = _b[0] * _tmp24
        _tmp37 = -_tmp17 + _tmp18
        _tmp38 = _b[0] * _tmp37
        _tmp39 = _b[1] * _tmp37
        _tmp40 = -_b[1] * _tmp16 + _tmp38

        # Output terms
        _res = [0.0] * 4
        _res[0] = _tmp6
        _res[1] = _tmp7
        _res[2] = _a[0] * _tmp11 - _a[0] * _tmp9 + _a[1] * _tmp10 - _a[1] * _tmp8
        _res[3] = _a[0] * _tmp10 - _a[0] * _tmp8 - _a[1] * _tmp11 + _a[1] * _tmp9
        _res_D_a = numpy.zeros((3, 3))
        _res_D_a[0, 0] = _a[1] * _tmp16 + _tmp20
        _res_D_a[0, 1] = _a[0] * _tmp16 - _tmp21
        _res_D_a[0, 2] = _a[0] * (_tmp26 * _tmp7 + _tmp30 * _tmp6) - _a[1] * (
            _tmp32 * _tmp7 + _tmp33 * _tmp6
        )
        _res_D_a[1, 0] = _a[0] * _tmp34 + _tmp21
        _res_D_a[1, 1] = -_a[1] * _tmp34 + _tmp20
        _res_D_a[1, 2] = _a[0] * (_tmp26 * _tmp6 - _tmp30 * _tmp7) - _a[1] * (
            _tmp32 * _tmp6 - _tmp33 * _tmp7
        )
        _res_D_a[2, 0] = 0
        _res_D_a[2, 1] = 0
        _res_D_a[2, 2] = _a[0] * (
            _tmp6 * (_b[0] * _tmp23 + _tmp35 - _tmp5) - _tmp7 * (-_b[1] * _tmp23 - _tmp36 + _tmp4)
        ) - _a[1] * (
            _tmp6 * (-_b[1] * _tmp31 + _tmp36 + _tmp4) - _tmp7 * (-_b[0] * _tmp31 + _tmp35 + _tmp5)
        )
        _res_D_b = numpy.zeros((3, 3))
        _res_D_b[0, 0] = _b[1] * _tmp34 + _tmp38
        _res_D_b[0, 1] = _b[0] * _tmp34 - _tmp39
        _res_D_b[0, 2] = 0
        _res_D_b[1, 0] = _b[0] * _tmp16 + _tmp39
        _res_D_b[1, 1] = _tmp40
        _res_D_b[1, 2] = 0
        _res_D_b[2, 0] = 0
        _res_D_b[2, 1] = 0
        _res_D_b[2, 2] = _tmp40
        return _res, _res_D_a, _res_D_b
