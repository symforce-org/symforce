import numpy


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.rot3.Rot3'>.
    """

    @staticmethod
    def identity():

        # Input arrays

        # Intermediate terms

        # Output terms
        _res = [0.0] * 4
        _res[0] = 0
        _res[1] = 0
        _res[2] = 0
        _res[3] = 1
        return _res

    @staticmethod
    def inverse(a):

        # Input arrays
        _a = a.data

        # Intermediate terms

        # Output terms
        _res = [0.0] * 4
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        return _res

    @staticmethod
    def compose(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms

        # Output terms
        _res = [0.0] * 4
        _res[0] = _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0]
        _res[1] = -_a[0] * _b[2] + _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1]
        _res[2] = _a[0] * _b[1] - _a[1] * _b[0] + _a[2] * _b[3] + _a[3] * _b[2]
        _res[3] = -_a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] + _a[3] * _b[3]
        return _res

    @staticmethod
    def between(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms

        # Output terms
        _res = [0.0] * 4
        _res[0] = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0]
        _res[1] = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1]
        _res[2] = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2]
        _res[3] = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3]
        return _res

    @staticmethod
    def inverse_with_jacobian(a):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = _a[2] ** 2
        _tmp1 = _a[0] ** 2
        _tmp2 = -_a[3] ** 2
        _tmp3 = _a[1] ** 2
        _tmp4 = _tmp2 + _tmp3
        _tmp5 = -2 * _a[0] * _a[1]
        _tmp6 = 2 * _a[2]
        _tmp7 = _a[3] * _tmp6
        _tmp8 = -_a[0] * _tmp6
        _tmp9 = 2 * _a[3]
        _tmp10 = _a[1] * _tmp9
        _tmp11 = _a[0] * _tmp9
        _tmp12 = -_a[1] * _tmp6

        # Output terms
        _res = [0.0] * 4
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        _res_D_a = numpy.zeros((3, 3))
        _res_D_a[0, 0] = _tmp0 - _tmp1 + _tmp4
        _res_D_a[0, 1] = _tmp5 + _tmp7
        _res_D_a[0, 2] = -_tmp10 + _tmp8
        _res_D_a[1, 0] = _tmp5 - _tmp7
        _res_D_a[1, 1] = _tmp0 + _tmp1 + _tmp2 - _tmp3
        _res_D_a[1, 2] = _tmp11 + _tmp12
        _res_D_a[2, 0] = _tmp10 + _tmp8
        _res_D_a[2, 1] = -_tmp11 + _tmp12
        _res_D_a[2, 2] = -_tmp0 + _tmp1 + _tmp4
        return _res, _res_D_a

    @staticmethod
    def compose_with_jacobians(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0]
        _tmp1 = -_a[0] * _b[2] + _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1]
        _tmp2 = _a[0] * _b[1] - _a[1] * _b[0] + _a[2] * _b[3] + _a[3] * _b[2]
        _tmp3 = -_a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] + _a[3] * _b[3]
        _tmp4 = 2 * _tmp2
        _tmp5 = _b[0] * _tmp4
        _tmp6 = 2 * _tmp3
        _tmp7 = _b[1] * _tmp6
        _tmp8 = 2 * _tmp1
        _tmp9 = _b[3] * _tmp8
        _tmp10 = 2 * _tmp0
        _tmp11 = _b[2] * _tmp10
        _tmp12 = _tmp11 - _tmp9
        _tmp13 = _tmp12 + _tmp5 - _tmp7
        _tmp14 = (1.0 / 2.0) * _a[1]
        _tmp15 = _b[2] * _tmp8
        _tmp16 = _b[0] * _tmp6
        _tmp17 = _b[3] * _tmp10
        _tmp18 = _b[1] * _tmp4
        _tmp19 = -_tmp17 + _tmp18
        _tmp20 = -_tmp15 + _tmp16 + _tmp19
        _tmp21 = (1.0 / 2.0) * _a[0]
        _tmp22 = _b[0] * _tmp8
        _tmp23 = _b[3] * _tmp4
        _tmp24 = _b[1] * _tmp10
        _tmp25 = _b[2] * _tmp6
        _tmp26 = _tmp22 + _tmp23 + _tmp24 + _tmp25
        _tmp27 = (1.0 / 2.0) * _a[2]
        _tmp28 = _b[0] * _tmp10
        _tmp29 = _b[1] * _tmp8
        _tmp30 = -_tmp29
        _tmp31 = _b[2] * _tmp4
        _tmp32 = -_tmp31
        _tmp33 = _b[3] * _tmp6
        _tmp34 = _tmp28 + _tmp30 + _tmp32 + _tmp33
        _tmp35 = (1.0 / 2.0) * _a[3]
        _tmp36 = _tmp12 - _tmp5 + _tmp7
        _tmp37 = _tmp15 + _tmp16 + _tmp17 + _tmp18
        _tmp38 = -_tmp28 + _tmp33
        _tmp39 = _tmp29 + _tmp32 + _tmp38
        _tmp40 = _tmp22 - _tmp23
        _tmp41 = _tmp24 - _tmp25 + _tmp40
        _tmp42 = -_tmp24 + _tmp25 + _tmp40
        _tmp43 = _tmp30 + _tmp31 + _tmp38
        _tmp44 = _tmp15 - _tmp16 + _tmp19
        _tmp45 = _tmp11 + _tmp5 + _tmp7 + _tmp9
        _tmp46 = _a[2] * _tmp10
        _tmp47 = _a[3] * _tmp8
        _tmp48 = _a[0] * _tmp4
        _tmp49 = _a[1] * _tmp6
        _tmp50 = _tmp46 - _tmp47 - _tmp48 + _tmp49
        _tmp51 = (1.0 / 2.0) * _b[1]
        _tmp52 = -_tmp50 * _tmp51
        _tmp53 = _a[0] * _tmp8
        _tmp54 = _a[3] * _tmp4
        _tmp55 = _a[1] * _tmp10
        _tmp56 = _a[2] * _tmp6
        _tmp57 = -_tmp53 + _tmp54 + _tmp55 - _tmp56
        _tmp58 = (1.0 / 2.0) * _b[2]
        _tmp59 = _a[0] * _tmp6
        _tmp60 = _a[3] * _tmp10
        _tmp61 = _a[1] * _tmp4
        _tmp62 = _a[2] * _tmp8
        _tmp63 = _tmp59 - _tmp60 + _tmp61 - _tmp62
        _tmp64 = (1.0 / 2.0) * _b[0]
        _tmp65 = _a[0] * _tmp10 + _a[1] * _tmp8 + _a[2] * _tmp4 + _a[3] * _tmp6
        _tmp66 = (1.0 / 2.0) * _b[3]
        _tmp67 = _tmp65 * _tmp66
        _tmp68 = -_tmp63 * _tmp64 + _tmp67
        _tmp69 = _tmp50 * _tmp64
        _tmp70 = _tmp58 * _tmp65
        _tmp71 = _tmp58 * _tmp63
        _tmp72 = _tmp51 * _tmp65
        _tmp73 = -_tmp59 + _tmp60 - _tmp61 + _tmp62
        _tmp74 = _tmp53 - _tmp54 - _tmp55 + _tmp56
        _tmp75 = -_tmp58 * _tmp74
        _tmp76 = _tmp64 * _tmp65
        _tmp77 = _tmp51 * _tmp74
        _tmp78 = -_tmp46 + _tmp47 + _tmp48 - _tmp49

        # Output terms
        _res = [0.0] * 4
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res[2] = _tmp2
        _res[3] = _tmp3
        _res_D_a = numpy.zeros((3, 3))
        _res_D_a[0, 0] = -_tmp13 * _tmp14 - _tmp20 * _tmp21 + _tmp26 * _tmp27 + _tmp34 * _tmp35
        _res_D_a[0, 1] = _tmp13 * _tmp21 - _tmp14 * _tmp20 + _tmp26 * _tmp35 - _tmp27 * _tmp34
        _res_D_a[0, 2] = _tmp13 * _tmp35 + _tmp14 * _tmp34 - _tmp20 * _tmp27 - _tmp21 * _tmp26
        _res_D_a[1, 0] = -_tmp14 * _tmp37 - _tmp21 * _tmp36 + _tmp27 * _tmp39 + _tmp35 * _tmp41
        _res_D_a[1, 1] = -_tmp14 * _tmp36 + _tmp21 * _tmp37 - _tmp27 * _tmp41 + _tmp35 * _tmp39
        _res_D_a[1, 2] = _tmp14 * _tmp41 - _tmp21 * _tmp39 - _tmp27 * _tmp36 + _tmp35 * _tmp37
        _res_D_a[2, 0] = -_tmp14 * _tmp43 - _tmp21 * _tmp42 + _tmp27 * _tmp44 + _tmp35 * _tmp45
        _res_D_a[2, 1] = -_tmp14 * _tmp42 + _tmp21 * _tmp43 - _tmp27 * _tmp45 + _tmp35 * _tmp44
        _res_D_a[2, 2] = _tmp14 * _tmp45 - _tmp21 * _tmp44 - _tmp27 * _tmp42 + _tmp35 * _tmp43
        _res_D_b = numpy.zeros((3, 3))
        _res_D_b[0, 0] = _tmp52 + _tmp57 * _tmp58 + _tmp68
        _res_D_b[0, 1] = -_tmp51 * _tmp63 + _tmp57 * _tmp66 + _tmp69 - _tmp70
        _res_D_b[0, 2] = _tmp50 * _tmp66 - _tmp57 * _tmp64 - _tmp71 + _tmp72
        _res_D_b[1, 0] = -_tmp51 * _tmp73 + _tmp66 * _tmp74 - _tmp69 + _tmp70
        _res_D_b[1, 1] = _tmp52 + _tmp64 * _tmp73 + _tmp67 + _tmp75
        _res_D_b[1, 2] = -_tmp50 * _tmp58 + _tmp66 * _tmp73 - _tmp76 + _tmp77
        _res_D_b[2, 0] = -_tmp64 * _tmp74 + _tmp66 * _tmp78 + _tmp71 - _tmp72
        _res_D_b[2, 1] = -_tmp58 * _tmp78 + _tmp63 * _tmp66 + _tmp76 - _tmp77
        _res_D_b[2, 2] = _tmp51 * _tmp78 + _tmp68 + _tmp75
        return _res, _res_D_a, _res_D_b

    @staticmethod
    def between_with_jacobians(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0]
        _tmp1 = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1]
        _tmp2 = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2]
        _tmp3 = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3]
        _tmp4 = 2 * _tmp2
        _tmp5 = _b[1] * _tmp4
        _tmp6 = 2 * _tmp3
        _tmp7 = _b[0] * _tmp6
        _tmp8 = 2 * _tmp0
        _tmp9 = _b[3] * _tmp8
        _tmp10 = 2 * _b[2]
        _tmp11 = -_tmp1 * _tmp10
        _tmp12 = _tmp11 - _tmp9
        _tmp13 = _tmp12 + _tmp5 + _tmp7
        _tmp14 = (1.0 / 2.0) * _a[0]
        _tmp15 = 2 * _tmp1
        _tmp16 = _b[3] * _tmp15
        _tmp17 = _b[1] * _tmp6
        _tmp18 = _tmp0 * _tmp10
        _tmp19 = -_b[0] * _tmp4
        _tmp20 = -_tmp18 + _tmp19
        _tmp21 = _tmp16 + _tmp17 + _tmp20
        _tmp22 = (1.0 / 2.0) * _a[1]
        _tmp23 = _b[0] * _tmp15
        _tmp24 = -_tmp23
        _tmp25 = -_b[1] * _tmp8
        _tmp26 = _b[3] * _tmp4
        _tmp27 = -_tmp26
        _tmp28 = _tmp10 * _tmp3
        _tmp29 = _tmp24 + _tmp25 + _tmp27 - _tmp28
        _tmp30 = (1.0 / 2.0) * _a[2]
        _tmp31 = _b[1] * _tmp15
        _tmp32 = _tmp10 * _tmp2
        _tmp33 = _b[0] * _tmp8
        _tmp34 = -_b[3] * _tmp6
        _tmp35 = _tmp31 + _tmp32 - _tmp33 + _tmp34
        _tmp36 = (1.0 / 2.0) * _a[3]
        _tmp37 = -_tmp16
        _tmp38 = _tmp17 + _tmp18 + _tmp19 + _tmp37
        _tmp39 = -_tmp5
        _tmp40 = _tmp12 + _tmp39 - _tmp7
        _tmp41 = _tmp33 + _tmp34
        _tmp42 = -_tmp31 + _tmp32 + _tmp41
        _tmp43 = _tmp25 + _tmp28
        _tmp44 = _tmp24 + _tmp26 + _tmp43
        _tmp45 = _tmp23 + _tmp27 + _tmp43
        _tmp46 = _tmp31 - _tmp32 + _tmp41
        _tmp47 = _tmp11 + _tmp39 + _tmp7 + _tmp9
        _tmp48 = -_tmp17 + _tmp20 + _tmp37
        _tmp49 = _a[2] * _tmp8
        _tmp50 = _a[3] * _tmp15
        _tmp51 = _a[1] * _tmp6
        _tmp52 = _a[0] * _tmp4
        _tmp53 = (
            -1.0 / 2.0 * _tmp49 - 1.0 / 2.0 * _tmp50 - 1.0 / 2.0 * _tmp51 + (1.0 / 2.0) * _tmp52
        )
        _tmp54 = -_b[1] * _tmp53
        _tmp55 = _a[1] * _tmp8
        _tmp56 = _a[0] * _tmp15
        _tmp57 = _a[3] * _tmp4
        _tmp58 = _a[2] * _tmp6
        _tmp59 = -_tmp55 + _tmp56 + _tmp57 + _tmp58
        _tmp60 = (1.0 / 2.0) * _b[2]
        _tmp61 = _a[3] * _tmp8
        _tmp62 = _a[1] * _tmp4
        _tmp63 = _a[2] * _tmp15
        _tmp64 = _a[0] * _tmp6
        _tmp65 = -_tmp61 - _tmp62 + _tmp63 - _tmp64
        _tmp66 = (1.0 / 2.0) * _b[0]
        _tmp67 = -_a[0] * _tmp8 - _a[1] * _tmp15 - _a[2] * _tmp4 + _a[3] * _tmp6
        _tmp68 = (1.0 / 2.0) * _b[3]
        _tmp69 = _tmp67 * _tmp68
        _tmp70 = -_tmp65 * _tmp66 + _tmp69
        _tmp71 = (1.0 / 2.0) * _b[1]
        _tmp72 = _b[0] * _tmp53
        _tmp73 = _tmp60 * _tmp67
        _tmp74 = _tmp60 * _tmp65
        _tmp75 = _tmp67 * _tmp71
        _tmp76 = _tmp61 + _tmp62 - _tmp63 + _tmp64
        _tmp77 = _tmp55 - _tmp56 - _tmp57 - _tmp58
        _tmp78 = -_tmp60 * _tmp77
        _tmp79 = _tmp66 * _tmp67
        _tmp80 = _tmp71 * _tmp77
        _tmp81 = _tmp49 + _tmp50 + _tmp51 - _tmp52

        # Output terms
        _res = [0.0] * 4
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res[2] = _tmp2
        _res[3] = _tmp3
        _res_D_a = numpy.zeros((3, 3))
        _res_D_a[0, 0] = -_tmp13 * _tmp14 - _tmp21 * _tmp22 + _tmp29 * _tmp30 + _tmp35 * _tmp36
        _res_D_a[0, 1] = -_tmp13 * _tmp22 + _tmp14 * _tmp21 + _tmp29 * _tmp36 - _tmp30 * _tmp35
        _res_D_a[0, 2] = -_tmp13 * _tmp30 - _tmp14 * _tmp29 + _tmp21 * _tmp36 + _tmp22 * _tmp35
        _res_D_a[1, 0] = -_tmp14 * _tmp38 - _tmp22 * _tmp40 + _tmp30 * _tmp42 + _tmp36 * _tmp44
        _res_D_a[1, 1] = _tmp14 * _tmp40 - _tmp22 * _tmp38 - _tmp30 * _tmp44 + _tmp36 * _tmp42
        _res_D_a[1, 2] = -_tmp14 * _tmp42 + _tmp22 * _tmp44 - _tmp30 * _tmp38 + _tmp36 * _tmp40
        _res_D_a[2, 0] = -_tmp14 * _tmp45 - _tmp22 * _tmp46 + _tmp30 * _tmp47 + _tmp36 * _tmp48
        _res_D_a[2, 1] = _tmp14 * _tmp46 - _tmp22 * _tmp45 - _tmp30 * _tmp48 + _tmp36 * _tmp47
        _res_D_a[2, 2] = -_tmp14 * _tmp47 + _tmp22 * _tmp48 - _tmp30 * _tmp45 + _tmp36 * _tmp46
        _res_D_b = numpy.zeros((3, 3))
        _res_D_b[0, 0] = _tmp54 + _tmp59 * _tmp60 + _tmp70
        _res_D_b[0, 1] = _tmp59 * _tmp68 - _tmp65 * _tmp71 + _tmp72 - _tmp73
        _res_D_b[0, 2] = _b[3] * _tmp53 - _tmp59 * _tmp66 - _tmp74 + _tmp75
        _res_D_b[1, 0] = _tmp68 * _tmp77 - _tmp71 * _tmp76 - _tmp72 + _tmp73
        _res_D_b[1, 1] = _tmp54 + _tmp66 * _tmp76 + _tmp69 + _tmp78
        _res_D_b[1, 2] = -_b[2] * _tmp53 + _tmp68 * _tmp76 - _tmp79 + _tmp80
        _res_D_b[2, 0] = -_tmp66 * _tmp77 + _tmp68 * _tmp81 + _tmp74 - _tmp75
        _res_D_b[2, 1] = -_tmp60 * _tmp81 + _tmp65 * _tmp68 + _tmp79 - _tmp80
        _res_D_b[2, 2] = _tmp70 + _tmp71 * _tmp81 + _tmp78
        return _res, _res_D_a, _res_D_b
