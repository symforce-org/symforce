import numpy


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.pose3.Pose3'>.
    """

    @staticmethod
    def identity():

        # Input arrays

        # Intermediate terms

        # Output terms
        _res = [0.0] * 7
        _res[0] = 0
        _res[1] = 0
        _res[2] = 0
        _res[3] = 1
        _res[4] = 0
        _res[5] = 0
        _res[6] = 0
        return _res

    @staticmethod
    def inverse(a):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = 2 * _a[1] * _a[3]
        _tmp1 = 2 * _a[0]
        _tmp2 = _a[2] * _tmp1
        _tmp3 = 2 * _a[2]
        _tmp4 = _a[3] * _tmp3
        _tmp5 = _a[1] * _tmp1
        _tmp6 = -2 * _a[1] ** 2
        _tmp7 = -2 * _a[2] ** 2 + 1
        _tmp8 = _a[3] * _tmp1
        _tmp9 = _a[1] * _tmp3
        _tmp10 = -2 * _a[0] ** 2

        # Output terms
        _res = [0.0] * 7
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        _res[4] = -_a[4] * (_tmp6 + _tmp7) - _a[5] * (_tmp4 + _tmp5) - _a[6] * (-_tmp0 + _tmp2)
        _res[5] = -_a[4] * (-_tmp4 + _tmp5) - _a[5] * (_tmp10 + _tmp7) - _a[6] * (_tmp8 + _tmp9)
        _res[6] = -_a[4] * (_tmp0 + _tmp2) - _a[5] * (-_tmp8 + _tmp9) - _a[6] * (_tmp10 + _tmp6 + 1)
        return _res

    @staticmethod
    def compose(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = 2 * _a[3]
        _tmp1 = _a[1] * _tmp0
        _tmp2 = 2 * _a[2]
        _tmp3 = _a[0] * _tmp2
        _tmp4 = _a[2] * _tmp0
        _tmp5 = 2 * _a[0] * _a[1]
        _tmp6 = -2 * _a[2] ** 2
        _tmp7 = -2 * _a[1] ** 2
        _tmp8 = _a[0] * _tmp0
        _tmp9 = _a[1] * _tmp2
        _tmp10 = -2 * _a[0] ** 2 + 1

        # Output terms
        _res = [0.0] * 7
        _res[0] = _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0]
        _res[1] = -_a[0] * _b[2] + _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1]
        _res[2] = _a[0] * _b[1] - _a[1] * _b[0] + _a[2] * _b[3] + _a[3] * _b[2]
        _res[3] = -_a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] + _a[3] * _b[3]
        _res[4] = (
            _a[4] + _b[4] * (_tmp6 + _tmp7 + 1) + _b[5] * (-_tmp4 + _tmp5) + _b[6] * (_tmp1 + _tmp3)
        )
        _res[5] = (
            _a[5] + _b[4] * (_tmp4 + _tmp5) + _b[5] * (_tmp10 + _tmp6) + _b[6] * (-_tmp8 + _tmp9)
        )
        _res[6] = (
            _a[6] + _b[4] * (-_tmp1 + _tmp3) + _b[5] * (_tmp8 + _tmp9) + _b[6] * (_tmp10 + _tmp7)
        )
        return _res

    @staticmethod
    def between(a, b):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = 2 * _a[1] * _a[3]
        _tmp1 = 2 * _a[0]
        _tmp2 = _a[2] * _tmp1
        _tmp3 = -_tmp0 + _tmp2
        _tmp4 = 2 * _a[2]
        _tmp5 = _a[3] * _tmp4
        _tmp6 = _a[1] * _tmp1
        _tmp7 = _tmp5 + _tmp6
        _tmp8 = -2 * _a[2] ** 2
        _tmp9 = -2 * _a[1] ** 2
        _tmp10 = _tmp8 + _tmp9 + 1
        _tmp11 = _a[3] * _tmp1
        _tmp12 = _a[1] * _tmp4
        _tmp13 = _tmp11 + _tmp12
        _tmp14 = -2 * _a[0] ** 2 + 1
        _tmp15 = _tmp14 + _tmp8
        _tmp16 = -_tmp5 + _tmp6
        _tmp17 = _tmp14 + _tmp9
        _tmp18 = -_tmp11 + _tmp12
        _tmp19 = _tmp0 + _tmp2

        # Output terms
        _res = [0.0] * 7
        _res[0] = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0]
        _res[1] = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1]
        _res[2] = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2]
        _res[3] = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3]
        _res[4] = (
            -_a[4] * _tmp10
            - _a[5] * _tmp7
            - _a[6] * _tmp3
            + _b[4] * _tmp10
            + _b[5] * _tmp7
            + _b[6] * _tmp3
        )
        _res[5] = (
            -_a[4] * _tmp16
            - _a[5] * _tmp15
            - _a[6] * _tmp13
            + _b[4] * _tmp16
            + _b[5] * _tmp15
            + _b[6] * _tmp13
        )
        _res[6] = (
            -_a[4] * _tmp19
            - _a[5] * _tmp18
            - _a[6] * _tmp17
            + _b[4] * _tmp19
            + _b[5] * _tmp18
            + _b[6] * _tmp17
        )
        return _res

    @staticmethod
    def inverse_with_jacobian(a):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = 2 * _a[3]
        _tmp1 = _a[1] * _tmp0
        _tmp2 = -_tmp1
        _tmp3 = 2 * _a[0]
        _tmp4 = _a[2] * _tmp3
        _tmp5 = _tmp2 + _tmp4
        _tmp6 = _a[2] * _tmp0
        _tmp7 = _a[1] * _tmp3
        _tmp8 = _tmp6 + _tmp7
        _tmp9 = _a[1] ** 2
        _tmp10 = -2 * _tmp9
        _tmp11 = _a[2] ** 2
        _tmp12 = -2 * _tmp11 + 1
        _tmp13 = _tmp10 + _tmp12
        _tmp14 = _a[3] * _tmp3
        _tmp15 = 2 * _a[2]
        _tmp16 = _a[1] * _tmp15
        _tmp17 = _tmp14 + _tmp16
        _tmp18 = _a[0] ** 2
        _tmp19 = -2 * _tmp18
        _tmp20 = _tmp12 + _tmp19
        _tmp21 = -_tmp6
        _tmp22 = _tmp21 + _tmp7
        _tmp23 = _tmp10 + _tmp19 + 1
        _tmp24 = -_tmp14
        _tmp25 = _tmp16 + _tmp24
        _tmp26 = _tmp1 + _tmp4
        _tmp27 = -_a[3] ** 2
        _tmp28 = _tmp11 + _tmp27
        _tmp29 = -_tmp7
        _tmp30 = -_tmp4
        _tmp31 = -_tmp16
        _tmp32 = _a[4] * _tmp3
        _tmp33 = 2 * _a[1]
        _tmp34 = _a[5] * _tmp33
        _tmp35 = _tmp32 + _tmp34
        _tmp36 = _a[4] * _tmp0
        _tmp37 = 4 * _a[5]
        _tmp38 = _a[6] * _tmp33
        _tmp39 = -_a[2] * _tmp37 - _tmp36 + _tmp38
        _tmp40 = 4 * _a[4]
        _tmp41 = _a[5] * _tmp0
        _tmp42 = _a[6] * _tmp3
        _tmp43 = -_a[2] * _tmp40 + _tmp41 + _tmp42
        _tmp44 = -_tmp13 * _tmp43 - _tmp22 * _tmp39 - _tmp26 * _tmp35
        _tmp45 = (1.0 / 2.0) * _a[1]
        _tmp46 = _a[6] * _tmp15
        _tmp47 = _tmp32 + _tmp46
        _tmp48 = _a[5] * _tmp15
        _tmp49 = 4 * _a[6]
        _tmp50 = -_a[1] * _tmp49 + _tmp36 + _tmp48
        _tmp51 = _a[5] * _tmp3
        _tmp52 = _a[6] * _tmp0
        _tmp53 = -_a[1] * _tmp40 + _tmp51 - _tmp52
        _tmp54 = -_tmp13 * _tmp53 - _tmp22 * _tmp47 - _tmp26 * _tmp50
        _tmp55 = (1.0 / 2.0) * _a[2]
        _tmp56 = _a[4] * _tmp15
        _tmp57 = _tmp42 - _tmp56
        _tmp58 = _a[4] * _tmp33
        _tmp59 = -_tmp51 + _tmp58
        _tmp60 = -_tmp38 + _tmp48
        _tmp61 = -_tmp13 * _tmp60 - _tmp22 * _tmp57 - _tmp26 * _tmp59
        _tmp62 = (1.0 / 2.0) * _a[0]
        _tmp63 = -_a[0] * _tmp49 - _tmp41 + _tmp56
        _tmp64 = -_a[0] * _tmp37 + _tmp52 + _tmp58
        _tmp65 = _tmp34 + _tmp46
        _tmp66 = -_tmp13 * _tmp65 - _tmp22 * _tmp64 - _tmp26 * _tmp63
        _tmp67 = (1.0 / 2.0) * _a[3]
        _tmp68 = -_tmp13 * _tmp5 - _tmp17 * _tmp22 - _tmp23 * _tmp26
        _tmp69 = -_tmp13 * _tmp8 - _tmp20 * _tmp22 - _tmp25 * _tmp26
        _tmp70 = -(_tmp13 ** 2) - _tmp22 ** 2 - _tmp26 ** 2
        _tmp71 = -_tmp20 * _tmp57 - _tmp25 * _tmp59 - _tmp60 * _tmp8
        _tmp72 = (1.0 / 2.0) * _tmp71
        _tmp73 = -_tmp20 * _tmp39 - _tmp25 * _tmp35 - _tmp43 * _tmp8
        _tmp74 = -_tmp20 * _tmp47 - _tmp25 * _tmp50 - _tmp53 * _tmp8
        _tmp75 = -_tmp20 * _tmp64 - _tmp25 * _tmp63 - _tmp65 * _tmp8
        _tmp76 = (1.0 / 2.0) * _tmp75
        _tmp77 = -(_tmp20 ** 2) - _tmp25 ** 2 - _tmp8 ** 2
        _tmp78 = -_tmp17 * _tmp20 - _tmp23 * _tmp25 - _tmp5 * _tmp8
        _tmp79 = (
            -1.0 / 2.0 * _tmp17 * _tmp57 - 1.0 / 2.0 * _tmp23 * _tmp59 - 1.0 / 2.0 * _tmp5 * _tmp60
        )
        _tmp80 = -_tmp17 * _tmp39 - _tmp23 * _tmp35 - _tmp43 * _tmp5
        _tmp81 = (
            -1.0 / 2.0 * _tmp17 * _tmp47 - 1.0 / 2.0 * _tmp23 * _tmp50 - 1.0 / 2.0 * _tmp5 * _tmp53
        )
        _tmp82 = -_tmp17 * _tmp64 - _tmp23 * _tmp63 - _tmp5 * _tmp65
        _tmp83 = (1.0 / 2.0) * _tmp82
        _tmp84 = (1.0 / 2.0) * _tmp80
        _tmp85 = -(_tmp17 ** 2) - _tmp23 ** 2 - _tmp5 ** 2

        # Output terms
        _res = [0.0] * 7
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        _res[4] = -_a[4] * _tmp13 - _a[5] * _tmp8 - _a[6] * _tmp5
        _res[5] = -_a[4] * _tmp22 - _a[5] * _tmp20 - _a[6] * _tmp17
        _res[6] = -_a[4] * _tmp26 - _a[5] * _tmp25 - _a[6] * _tmp23
        _res_D_a = numpy.zeros((6, 6))
        _res_D_a[0, 0] = -_tmp18 + _tmp28 + _tmp9
        _res_D_a[0, 1] = _tmp29 + _tmp6
        _res_D_a[0, 2] = _tmp2 + _tmp30
        _res_D_a[0, 3] = 0
        _res_D_a[0, 4] = 0
        _res_D_a[0, 5] = 0
        _res_D_a[1, 0] = _tmp21 + _tmp29
        _res_D_a[1, 1] = _tmp18 + _tmp28 - _tmp9
        _res_D_a[1, 2] = _tmp14 + _tmp31
        _res_D_a[1, 3] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 0] = _tmp1 + _tmp30
        _res_D_a[2, 1] = _tmp24 + _tmp31
        _res_D_a[2, 2] = -_tmp11 + _tmp18 + _tmp27 + _tmp9
        _res_D_a[2, 3] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 0] = -_tmp44 * _tmp45 + _tmp54 * _tmp55 - _tmp61 * _tmp62 + _tmp66 * _tmp67
        _res_D_a[3, 1] = _tmp44 * _tmp62 - _tmp45 * _tmp61 + _tmp54 * _tmp67 - _tmp55 * _tmp66
        _res_D_a[3, 2] = _tmp44 * _tmp67 + _tmp45 * _tmp66 - _tmp54 * _tmp62 - _tmp55 * _tmp61
        _res_D_a[3, 3] = _tmp13 * _tmp70 + _tmp5 * _tmp68 + _tmp69 * _tmp8
        _res_D_a[3, 4] = _tmp17 * _tmp68 + _tmp20 * _tmp69 + _tmp22 * _tmp70
        _res_D_a[3, 5] = _tmp23 * _tmp68 + _tmp25 * _tmp69 + _tmp26 * _tmp70
        _res_D_a[4, 0] = -_a[0] * _tmp72 + _a[3] * _tmp76 - _tmp45 * _tmp73 + _tmp55 * _tmp74
        _res_D_a[4, 1] = -_a[2] * _tmp76 - _tmp45 * _tmp71 + _tmp62 * _tmp73 + _tmp67 * _tmp74
        _res_D_a[4, 2] = -_a[2] * _tmp72 + _tmp45 * _tmp75 - _tmp62 * _tmp74 + _tmp67 * _tmp73
        _res_D_a[4, 3] = _tmp13 * _tmp69 + _tmp5 * _tmp78 + _tmp77 * _tmp8
        _res_D_a[4, 4] = _tmp17 * _tmp78 + _tmp20 * _tmp77 + _tmp22 * _tmp69
        _res_D_a[4, 5] = _tmp23 * _tmp78 + _tmp25 * _tmp77 + _tmp26 * _tmp69
        _res_D_a[5, 0] = -_a[0] * _tmp79 + _a[2] * _tmp81 + _a[3] * _tmp83 - _tmp45 * _tmp80
        _res_D_a[5, 1] = _a[0] * _tmp84 - _a[1] * _tmp79 - _a[2] * _tmp83 + _a[3] * _tmp81
        _res_D_a[5, 2] = -_a[0] * _tmp81 - _a[2] * _tmp79 + _a[3] * _tmp84 + _tmp45 * _tmp82
        _res_D_a[5, 3] = _tmp13 * _tmp68 + _tmp5 * _tmp85 + _tmp78 * _tmp8
        _res_D_a[5, 4] = _tmp17 * _tmp85 + _tmp20 * _tmp78 + _tmp22 * _tmp68
        _res_D_a[5, 5] = _tmp23 * _tmp85 + _tmp25 * _tmp78 + _tmp26 * _tmp68
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
        _tmp4 = 2 * _a[3]
        _tmp5 = _a[1] * _tmp4
        _tmp6 = 2 * _a[2]
        _tmp7 = _a[0] * _tmp6
        _tmp8 = _tmp5 + _tmp7
        _tmp9 = _a[2] * _tmp4
        _tmp10 = 2 * _a[1]
        _tmp11 = _a[0] * _tmp10
        _tmp12 = _tmp11 - _tmp9
        _tmp13 = -2 * _a[1] ** 2
        _tmp14 = -2 * _a[2] ** 2 + 1
        _tmp15 = _tmp13 + _tmp14
        _tmp16 = _a[0] * _tmp4
        _tmp17 = _a[2] * _tmp10
        _tmp18 = -_tmp16 + _tmp17
        _tmp19 = -2 * _a[0] ** 2
        _tmp20 = _tmp14 + _tmp19
        _tmp21 = _tmp11 + _tmp9
        _tmp22 = _tmp13 + _tmp19 + 1
        _tmp23 = _tmp16 + _tmp17
        _tmp24 = -_tmp5 + _tmp7
        _tmp25 = 2 * _tmp0
        _tmp26 = _b[3] * _tmp25
        _tmp27 = -_tmp26
        _tmp28 = 2 * _tmp1
        _tmp29 = _b[2] * _tmp28
        _tmp30 = 2 * _b[1]
        _tmp31 = _tmp2 * _tmp30
        _tmp32 = 2 * _b[0]
        _tmp33 = _tmp3 * _tmp32
        _tmp34 = _tmp27 - _tmp29 + _tmp31 + _tmp33
        _tmp35 = (1.0 / 2.0) * _a[0]
        _tmp36 = 2 * _b[3]
        _tmp37 = _tmp2 * _tmp36
        _tmp38 = _b[1] * _tmp25
        _tmp39 = 2 * _b[2] * _tmp3
        _tmp40 = _tmp1 * _tmp32
        _tmp41 = _tmp39 + _tmp40
        _tmp42 = _tmp37 + _tmp38 + _tmp41
        _tmp43 = (1.0 / 2.0) * _a[2]
        _tmp44 = _b[2] * _tmp25
        _tmp45 = _tmp1 * _tmp36
        _tmp46 = -_tmp45
        _tmp47 = _tmp2 * _tmp32
        _tmp48 = _tmp3 * _tmp30
        _tmp49 = _tmp44 + _tmp46 + _tmp47 - _tmp48
        _tmp50 = (1.0 / 2.0) * _a[1]
        _tmp51 = _b[0] * _tmp25
        _tmp52 = 2 * _tmp2
        _tmp53 = _b[2] * _tmp52
        _tmp54 = -_tmp53
        _tmp55 = _tmp3 * _tmp36
        _tmp56 = _tmp1 * _tmp30
        _tmp57 = _tmp55 - _tmp56
        _tmp58 = _tmp51 + _tmp54 + _tmp57
        _tmp59 = (1.0 / 2.0) * _a[3]
        _tmp60 = _tmp44 + _tmp48
        _tmp61 = _tmp46 - _tmp47 + _tmp60
        _tmp62 = _tmp29 + _tmp31
        _tmp63 = _tmp26 + _tmp33 + _tmp62
        _tmp64 = -_tmp51
        _tmp65 = _tmp54 + _tmp55 + _tmp56 + _tmp64
        _tmp66 = -_tmp37
        _tmp67 = _tmp38 - _tmp39 + _tmp40 + _tmp66
        _tmp68 = -_tmp38 + _tmp41 + _tmp66
        _tmp69 = _tmp53 + _tmp57 + _tmp64
        _tmp70 = _tmp27 - _tmp33 + _tmp62
        _tmp71 = _tmp45 + _tmp47 + _tmp60
        _tmp72 = 2 * _a[0]
        _tmp73 = _b[4] * _tmp72
        _tmp74 = _b[5] * _tmp10
        _tmp75 = _tmp73 + _tmp74
        _tmp76 = _tmp28 * _tmp3
        _tmp77 = _tmp2 * _tmp25
        _tmp78 = -_tmp76 + _tmp77
        _tmp79 = _b[4] * _tmp4
        _tmp80 = 4 * _b[5]
        _tmp81 = _b[6] * _tmp10
        _tmp82 = -_a[2] * _tmp80 + _tmp79 + _tmp81
        _tmp83 = _tmp3 * _tmp52
        _tmp84 = _tmp1 * _tmp25
        _tmp85 = _tmp83 + _tmp84
        _tmp86 = -2 * _tmp2 ** 2
        _tmp87 = -2 * _tmp1 ** 2
        _tmp88 = _tmp86 + _tmp87 + 1
        _tmp89 = 4 * _b[4]
        _tmp90 = _b[5] * _tmp4
        _tmp91 = _b[6] * _tmp72
        _tmp92 = -_a[2] * _tmp89 - _tmp90 + _tmp91
        _tmp93 = _tmp75 * _tmp78 + _tmp82 * _tmp85 + _tmp88 * _tmp92
        _tmp94 = _b[5] * _tmp6
        _tmp95 = 4 * _b[6]
        _tmp96 = -_a[1] * _tmp95 - _tmp79 + _tmp94
        _tmp97 = _b[6] * _tmp6
        _tmp98 = _tmp73 + _tmp97
        _tmp99 = _b[5] * _tmp72
        _tmp100 = _b[6] * _tmp4
        _tmp101 = -_a[1] * _tmp89 + _tmp100 + _tmp99
        _tmp102 = _tmp101 * _tmp88 + _tmp78 * _tmp96 + _tmp85 * _tmp98
        _tmp103 = _b[4] * _tmp6
        _tmp104 = _tmp103 - _tmp91
        _tmp105 = _b[4] * _tmp10
        _tmp106 = -_tmp105 + _tmp99
        _tmp107 = _tmp81 - _tmp94
        _tmp108 = _tmp104 * _tmp85 + _tmp106 * _tmp78 + _tmp107 * _tmp88
        _tmp109 = -_a[0] * _tmp95 + _tmp103 + _tmp90
        _tmp110 = -_a[0] * _tmp80 - _tmp100 + _tmp105
        _tmp111 = _tmp74 + _tmp97
        _tmp112 = (
            (1.0 / 2.0) * _tmp109 * _tmp78
            + (1.0 / 2.0) * _tmp110 * _tmp85
            + (1.0 / 2.0) * _tmp111 * _tmp88
        )
        _tmp113 = _tmp15 * _tmp88 + _tmp21 * _tmp85 + _tmp24 * _tmp78
        _tmp114 = _tmp12 * _tmp88 + _tmp20 * _tmp85 + _tmp23 * _tmp78
        _tmp115 = _tmp18 * _tmp85 + _tmp22 * _tmp78 + _tmp8 * _tmp88
        _tmp116 = _tmp25 * _tmp3
        _tmp117 = _tmp1 * _tmp52
        _tmp118 = _tmp116 + _tmp117
        _tmp119 = -2 * _tmp0 ** 2 + 1
        _tmp120 = _tmp119 + _tmp86
        _tmp121 = -_tmp83 + _tmp84
        _tmp122 = _tmp104 * _tmp120 + _tmp106 * _tmp118 + _tmp107 * _tmp121
        _tmp123 = _tmp118 * _tmp75 + _tmp120 * _tmp82 + _tmp121 * _tmp92
        _tmp124 = (
            (1.0 / 2.0) * _tmp101 * _tmp121
            + (1.0 / 2.0) * _tmp118 * _tmp96
            + (1.0 / 2.0) * _tmp120 * _tmp98
        )
        _tmp125 = _tmp109 * _tmp118 + _tmp110 * _tmp120 + _tmp111 * _tmp121
        _tmp126 = _tmp118 * _tmp24 + _tmp120 * _tmp21 + _tmp121 * _tmp15
        _tmp127 = _tmp118 * _tmp23 + _tmp12 * _tmp121 + _tmp120 * _tmp20
        _tmp128 = _tmp118 * _tmp22 + _tmp120 * _tmp18 + _tmp121 * _tmp8
        _tmp129 = _tmp119 + _tmp87
        _tmp130 = -_tmp116 + _tmp117
        _tmp131 = _tmp76 + _tmp77
        _tmp132 = _tmp129 * _tmp75 + _tmp130 * _tmp82 + _tmp131 * _tmp92
        _tmp133 = _tmp104 * _tmp130 + _tmp106 * _tmp129 + _tmp107 * _tmp131
        _tmp134 = _tmp101 * _tmp131 + _tmp129 * _tmp96 + _tmp130 * _tmp98
        _tmp135 = _tmp109 * _tmp129 + _tmp110 * _tmp130 + _tmp111 * _tmp131
        _tmp136 = _tmp129 * _tmp24 + _tmp130 * _tmp21 + _tmp131 * _tmp15
        _tmp137 = _tmp12 * _tmp131 + _tmp129 * _tmp23 + _tmp130 * _tmp20
        _tmp138 = _tmp129 * _tmp22 + _tmp130 * _tmp18 + _tmp131 * _tmp8
        _tmp139 = _tmp0 * _tmp4
        _tmp140 = _tmp1 * _tmp6
        _tmp141 = _tmp10 * _tmp2
        _tmp142 = _tmp3 * _tmp72
        _tmp143 = -_tmp139 - _tmp140 + _tmp141 + _tmp142
        _tmp144 = (1.0 / 2.0) * _b[0]
        _tmp145 = -_tmp143 * _tmp144
        _tmp146 = _a[2] * _tmp25
        _tmp147 = _tmp2 * _tmp72
        _tmp148 = _tmp1 * _tmp4
        _tmp149 = _tmp10 * _tmp3
        _tmp150 = _tmp146 - _tmp147 - _tmp148 + _tmp149
        _tmp151 = (1.0 / 2.0) * _b[1]
        _tmp152 = -_tmp150 * _tmp151
        _tmp153 = _tmp0 * _tmp10
        _tmp154 = _tmp1 * _tmp72
        _tmp155 = _tmp2 * _tmp4
        _tmp156 = _tmp3 * _tmp6
        _tmp157 = (
            (1.0 / 2.0) * _tmp153
            - 1.0 / 2.0 * _tmp154
            + (1.0 / 2.0) * _tmp155
            - 1.0 / 2.0 * _tmp156
        )
        _tmp158 = _a[0] * _tmp25 + _tmp1 * _tmp10 + _tmp2 * _tmp6 + _tmp3 * _tmp4
        _tmp159 = (1.0 / 2.0) * _b[3]
        _tmp160 = _tmp158 * _tmp159
        _tmp161 = _tmp144 * _tmp150
        _tmp162 = (1.0 / 2.0) * _b[2]
        _tmp163 = _tmp158 * _tmp162
        _tmp164 = _tmp143 * _tmp162
        _tmp165 = _tmp151 * _tmp158
        _tmp166 = _tmp139 + _tmp140 - _tmp141 - _tmp142
        _tmp167 = (
            -1.0 / 2.0 * _tmp153
            + (1.0 / 2.0) * _tmp154
            - 1.0 / 2.0 * _tmp155
            + (1.0 / 2.0) * _tmp156
        )
        _tmp168 = -_b[2] * _tmp167 + _tmp160
        _tmp169 = _tmp144 * _tmp158
        _tmp170 = _b[1] * _tmp167
        _tmp171 = -_tmp146 + _tmp147 + _tmp148 - _tmp149
        _tmp172 = _b[3] * _tmp30
        _tmp173 = _b[2] * _tmp32
        _tmp174 = -_tmp172 + _tmp173
        _tmp175 = _b[2] * _tmp36
        _tmp176 = _b[1] * _tmp32
        _tmp177 = _tmp175 + _tmp176
        _tmp178 = -2 * _b[2] ** 2
        _tmp179 = -2 * _b[1] ** 2 + 1
        _tmp180 = _tmp178 + _tmp179
        _tmp181 = _b[3] * _tmp32
        _tmp182 = _b[2] * _tmp30
        _tmp183 = _tmp181 + _tmp182
        _tmp184 = -2 * _b[0] ** 2
        _tmp185 = _tmp178 + _tmp184 + 1
        _tmp186 = -_tmp175 + _tmp176
        _tmp187 = -_tmp181 + _tmp182
        _tmp188 = _tmp179 + _tmp184
        _tmp189 = _tmp172 + _tmp173

        # Output terms
        _res = [0.0] * 7
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res[2] = _tmp2
        _res[3] = _tmp3
        _res[4] = _a[4] + _b[4] * _tmp15 + _b[5] * _tmp12 + _b[6] * _tmp8
        _res[5] = _a[5] + _b[4] * _tmp21 + _b[5] * _tmp20 + _b[6] * _tmp18
        _res[6] = _a[6] + _b[4] * _tmp24 + _b[5] * _tmp23 + _b[6] * _tmp22
        _res_D_a = numpy.zeros((6, 6))
        _res_D_a[0, 0] = -_tmp34 * _tmp35 + _tmp42 * _tmp43 - _tmp49 * _tmp50 + _tmp58 * _tmp59
        _res_D_a[0, 1] = -_tmp34 * _tmp50 + _tmp35 * _tmp49 + _tmp42 * _tmp59 - _tmp43 * _tmp58
        _res_D_a[0, 2] = -_tmp34 * _tmp43 - _tmp35 * _tmp42 + _tmp49 * _tmp59 + _tmp50 * _tmp58
        _res_D_a[0, 3] = 0
        _res_D_a[0, 4] = 0
        _res_D_a[0, 5] = 0
        _res_D_a[1, 0] = -_tmp35 * _tmp61 + _tmp43 * _tmp65 - _tmp50 * _tmp63 + _tmp59 * _tmp67
        _res_D_a[1, 1] = _tmp35 * _tmp63 - _tmp43 * _tmp67 - _tmp50 * _tmp61 + _tmp59 * _tmp65
        _res_D_a[1, 2] = -_tmp35 * _tmp65 - _tmp43 * _tmp61 + _tmp50 * _tmp67 + _tmp59 * _tmp63
        _res_D_a[1, 3] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 0] = -_tmp35 * _tmp68 + _tmp43 * _tmp70 - _tmp50 * _tmp69 + _tmp59 * _tmp71
        _res_D_a[2, 1] = _tmp35 * _tmp69 - _tmp43 * _tmp71 - _tmp50 * _tmp68 + _tmp59 * _tmp70
        _res_D_a[2, 2] = -_tmp35 * _tmp70 - _tmp43 * _tmp68 + _tmp50 * _tmp71 + _tmp59 * _tmp69
        _res_D_a[2, 3] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 0] = _a[3] * _tmp112 + _tmp102 * _tmp43 - _tmp108 * _tmp35 - _tmp50 * _tmp93
        _res_D_a[3, 1] = -_a[2] * _tmp112 + _tmp102 * _tmp59 - _tmp108 * _tmp50 + _tmp35 * _tmp93
        _res_D_a[3, 2] = _a[1] * _tmp112 - _tmp102 * _tmp35 - _tmp108 * _tmp43 + _tmp59 * _tmp93
        _res_D_a[3, 3] = _tmp113
        _res_D_a[3, 4] = _tmp114
        _res_D_a[3, 5] = _tmp115
        _res_D_a[4, 0] = _a[2] * _tmp124 - _tmp122 * _tmp35 - _tmp123 * _tmp50 + _tmp125 * _tmp59
        _res_D_a[4, 1] = _a[3] * _tmp124 - _tmp122 * _tmp50 + _tmp123 * _tmp35 - _tmp125 * _tmp43
        _res_D_a[4, 2] = -_a[0] * _tmp124 - _tmp122 * _tmp43 + _tmp123 * _tmp59 + _tmp125 * _tmp50
        _res_D_a[4, 3] = _tmp126
        _res_D_a[4, 4] = _tmp127
        _res_D_a[4, 5] = _tmp128
        _res_D_a[5, 0] = -_tmp132 * _tmp50 - _tmp133 * _tmp35 + _tmp134 * _tmp43 + _tmp135 * _tmp59
        _res_D_a[5, 1] = _tmp132 * _tmp35 - _tmp133 * _tmp50 + _tmp134 * _tmp59 - _tmp135 * _tmp43
        _res_D_a[5, 2] = _tmp132 * _tmp59 - _tmp133 * _tmp43 - _tmp134 * _tmp35 + _tmp135 * _tmp50
        _res_D_a[5, 3] = _tmp136
        _res_D_a[5, 4] = _tmp137
        _res_D_a[5, 5] = _tmp138
        _res_D_b = numpy.zeros((6, 6))
        _res_D_b[0, 0] = _b[2] * _tmp157 + _tmp145 + _tmp152 + _tmp160
        _res_D_b[0, 1] = _b[3] * _tmp157 - _tmp143 * _tmp151 + _tmp161 - _tmp163
        _res_D_b[0, 2] = -_b[0] * _tmp157 + _tmp150 * _tmp159 - _tmp164 + _tmp165
        _res_D_b[0, 3] = 0
        _res_D_b[0, 4] = 0
        _res_D_b[0, 5] = 0
        _res_D_b[1, 0] = _b[3] * _tmp167 - _tmp151 * _tmp166 - _tmp161 + _tmp163
        _res_D_b[1, 1] = _tmp144 * _tmp166 + _tmp152 + _tmp168
        _res_D_b[1, 2] = -_tmp150 * _tmp162 + _tmp159 * _tmp166 - _tmp169 + _tmp170
        _res_D_b[1, 3] = 0
        _res_D_b[1, 4] = 0
        _res_D_b[1, 5] = 0
        _res_D_b[2, 0] = -_b[0] * _tmp167 + _tmp159 * _tmp171 + _tmp164 - _tmp165
        _res_D_b[2, 1] = _tmp143 * _tmp159 - _tmp162 * _tmp171 + _tmp169 - _tmp170
        _res_D_b[2, 2] = _tmp145 + _tmp151 * _tmp171 + _tmp168
        _res_D_b[2, 3] = 0
        _res_D_b[2, 4] = 0
        _res_D_b[2, 5] = 0
        _res_D_b[3, 0] = 0
        _res_D_b[3, 1] = 0
        _res_D_b[3, 2] = 0
        _res_D_b[3, 3] = _tmp113 * _tmp180 + _tmp114 * _tmp177 + _tmp115 * _tmp174
        _res_D_b[3, 4] = _tmp113 * _tmp186 + _tmp114 * _tmp185 + _tmp115 * _tmp183
        _res_D_b[3, 5] = _tmp113 * _tmp189 + _tmp114 * _tmp187 + _tmp115 * _tmp188
        _res_D_b[4, 0] = 0
        _res_D_b[4, 1] = 0
        _res_D_b[4, 2] = 0
        _res_D_b[4, 3] = _tmp126 * _tmp180 + _tmp127 * _tmp177 + _tmp128 * _tmp174
        _res_D_b[4, 4] = _tmp126 * _tmp186 + _tmp127 * _tmp185 + _tmp128 * _tmp183
        _res_D_b[4, 5] = _tmp126 * _tmp189 + _tmp127 * _tmp187 + _tmp128 * _tmp188
        _res_D_b[5, 0] = 0
        _res_D_b[5, 1] = 0
        _res_D_b[5, 2] = 0
        _res_D_b[5, 3] = _tmp136 * _tmp180 + _tmp137 * _tmp177 + _tmp138 * _tmp174
        _res_D_b[5, 4] = _tmp136 * _tmp186 + _tmp137 * _tmp185 + _tmp138 * _tmp183
        _res_D_b[5, 5] = _tmp136 * _tmp189 + _tmp137 * _tmp187 + _tmp138 * _tmp188
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
        _tmp4 = 2 * _a[3]
        _tmp5 = _a[1] * _tmp4
        _tmp6 = -_tmp5
        _tmp7 = 2 * _a[0]
        _tmp8 = _a[2] * _tmp7
        _tmp9 = _tmp6 + _tmp8
        _tmp10 = 2 * _a[2]
        _tmp11 = _a[3] * _tmp10
        _tmp12 = _a[1] * _tmp7
        _tmp13 = _tmp11 + _tmp12
        _tmp14 = 2 * _a[1] ** 2
        _tmp15 = -_tmp14
        _tmp16 = 2 * _a[2] ** 2
        _tmp17 = -_tmp16 + 1
        _tmp18 = _tmp15 + _tmp17
        _tmp19 = _a[3] * _tmp7
        _tmp20 = _a[1] * _tmp10
        _tmp21 = _tmp19 + _tmp20
        _tmp22 = 2 * _a[0] ** 2
        _tmp23 = -_tmp22
        _tmp24 = _tmp17 + _tmp23
        _tmp25 = -_tmp11
        _tmp26 = _tmp12 + _tmp25
        _tmp27 = _tmp15 + _tmp23 + 1
        _tmp28 = -_tmp19
        _tmp29 = _tmp20 + _tmp28
        _tmp30 = _tmp5 + _tmp8
        _tmp31 = 2 * _tmp0
        _tmp32 = -_b[1] * _tmp31
        _tmp33 = 2 * _tmp1
        _tmp34 = _b[0] * _tmp33
        _tmp35 = -_tmp34
        _tmp36 = 2 * _tmp2
        _tmp37 = _b[3] * _tmp36
        _tmp38 = -_tmp37
        _tmp39 = 2 * _b[2]
        _tmp40 = _tmp3 * _tmp39
        _tmp41 = _tmp32 + _tmp35 + _tmp38 - _tmp40
        _tmp42 = (1.0 / 2.0) * _a[2]
        _tmp43 = _b[3] * _tmp31
        _tmp44 = -_tmp43
        _tmp45 = _b[1] * _tmp36
        _tmp46 = -_tmp1 * _tmp39
        _tmp47 = 2 * _b[0] * _tmp3
        _tmp48 = _tmp46 + _tmp47
        _tmp49 = _tmp44 + _tmp45 + _tmp48
        _tmp50 = (1.0 / 2.0) * _a[0]
        _tmp51 = _tmp0 * _tmp39
        _tmp52 = -_tmp51
        _tmp53 = _b[3] * _tmp33
        _tmp54 = -_b[0] * _tmp36
        _tmp55 = 2 * _b[1]
        _tmp56 = _tmp3 * _tmp55
        _tmp57 = _tmp52 + _tmp53 + _tmp54 + _tmp56
        _tmp58 = (1.0 / 2.0) * _a[1]
        _tmp59 = _b[1] * _tmp33
        _tmp60 = _b[0] * _tmp31
        _tmp61 = _tmp2 * _tmp39
        _tmp62 = 2 * _b[3]
        _tmp63 = -_tmp3 * _tmp62
        _tmp64 = _tmp61 + _tmp63
        _tmp65 = _tmp59 - _tmp60 + _tmp64
        _tmp66 = (1.0 / 2.0) * _a[3]
        _tmp67 = -_tmp53 + _tmp54
        _tmp68 = _tmp51 + _tmp56 + _tmp67
        _tmp69 = -_tmp45
        _tmp70 = _tmp44 + _tmp46 - _tmp47 + _tmp69
        _tmp71 = -_tmp59 + _tmp60 + _tmp64
        _tmp72 = _tmp32 + _tmp40
        _tmp73 = _tmp35 + _tmp37 + _tmp72
        _tmp74 = _tmp59 + _tmp60 - _tmp61 + _tmp63
        _tmp75 = _tmp34 + _tmp38 + _tmp72
        _tmp76 = _tmp43 + _tmp48 + _tmp69
        _tmp77 = _tmp52 - _tmp56 + _tmp67
        _tmp78 = _b[5] * _tmp7
        _tmp79 = _a[5] * _tmp7
        _tmp80 = 2 * _a[1]
        _tmp81 = -_a[4] * _tmp80 + _b[4] * _tmp80
        _tmp82 = -_tmp78 + _tmp79 + _tmp81
        _tmp83 = _tmp3 * _tmp33
        _tmp84 = _tmp2 * _tmp31
        _tmp85 = -_tmp83 + _tmp84
        _tmp86 = _b[4] * _tmp10
        _tmp87 = _a[4] * _tmp10
        _tmp88 = -_a[6] * _tmp7 + _b[6] * _tmp7
        _tmp89 = -_tmp86 + _tmp87 + _tmp88
        _tmp90 = _tmp3 * _tmp36
        _tmp91 = _tmp0 * _tmp33
        _tmp92 = _tmp90 + _tmp91
        _tmp93 = _b[6] * _tmp80
        _tmp94 = _a[6] * _tmp80
        _tmp95 = -_a[5] * _tmp10 + _b[5] * _tmp10
        _tmp96 = -_tmp93 + _tmp94 + _tmp95
        _tmp97 = -2 * _tmp2 ** 2
        _tmp98 = -2 * _tmp1 ** 2 + 1
        _tmp99 = _tmp97 + _tmp98
        _tmp100 = _tmp82 * _tmp85 + _tmp89 * _tmp92 + _tmp96 * _tmp99
        _tmp101 = _b[4] * _tmp4
        _tmp102 = 4 * _a[2]
        _tmp103 = _a[4] * _tmp4
        _tmp104 = _a[5] * _tmp102 - _b[5] * _tmp102 - _tmp101 + _tmp103 + _tmp93 - _tmp94
        _tmp105 = -_a[4] * _tmp7 + _b[4] * _tmp7
        _tmp106 = -_a[5] * _tmp80 + _b[5] * _tmp80
        _tmp107 = _tmp105 + _tmp106
        _tmp108 = _b[5] * _tmp4
        _tmp109 = _a[5] * _tmp4
        _tmp110 = _a[4] * _tmp102 - _b[4] * _tmp102 + _tmp108 - _tmp109 + _tmp88
        _tmp111 = _tmp104 * _tmp92 + _tmp107 * _tmp85 + _tmp110 * _tmp99
        _tmp112 = 4 * _a[6]
        _tmp113 = 4 * _a[1]
        _tmp114 = _a[1] * _tmp112 - _b[6] * _tmp113 + _tmp101 - _tmp103 + _tmp95
        _tmp115 = -_a[6] * _tmp10 + _b[6] * _tmp10
        _tmp116 = _tmp105 + _tmp115
        _tmp117 = _a[6] * _tmp4
        _tmp118 = _b[6] * _tmp4
        _tmp119 = _a[4] * _tmp113 - _b[4] * _tmp113 + _tmp117 - _tmp118 + _tmp78 - _tmp79
        _tmp120 = _tmp114 * _tmp85 + _tmp116 * _tmp92 + _tmp119 * _tmp99
        _tmp121 = 4 * _a[0]
        _tmp122 = _a[0] * _tmp112 - _b[6] * _tmp121 - _tmp108 + _tmp109 + _tmp86 - _tmp87
        _tmp123 = _a[5] * _tmp121 - _b[5] * _tmp121 - _tmp117 + _tmp118 + _tmp81
        _tmp124 = _tmp106 + _tmp115
        _tmp125 = (
            (1.0 / 2.0) * _tmp122 * _tmp85
            + (1.0 / 2.0) * _tmp123 * _tmp92
            + (1.0 / 2.0) * _tmp124 * _tmp99
        )
        _tmp126 = _tmp14 + _tmp22 - 1
        _tmp127 = -_tmp20
        _tmp128 = _tmp127 + _tmp28
        _tmp129 = -_tmp8
        _tmp130 = _tmp129 + _tmp5
        _tmp131 = _tmp126 * _tmp85 + _tmp128 * _tmp92 + _tmp130 * _tmp99
        _tmp132 = _tmp127 + _tmp19
        _tmp133 = _tmp16 - 1
        _tmp134 = _tmp133 + _tmp22
        _tmp135 = -_tmp12
        _tmp136 = _tmp135 + _tmp25
        _tmp137 = _tmp132 * _tmp85 + _tmp134 * _tmp92 + _tmp136 * _tmp99
        _tmp138 = _tmp129 + _tmp6
        _tmp139 = _tmp11 + _tmp135
        _tmp140 = _tmp133 + _tmp14
        _tmp141 = _tmp138 * _tmp85 + _tmp139 * _tmp92 + _tmp140 * _tmp99
        _tmp142 = _tmp3 * _tmp31
        _tmp143 = _tmp2 * _tmp33
        _tmp144 = _tmp142 + _tmp143
        _tmp145 = -2 * _tmp0 ** 2
        _tmp146 = _tmp145 + _tmp97 + 1
        _tmp147 = -_tmp90 + _tmp91
        _tmp148 = _tmp144 * _tmp82 + _tmp146 * _tmp89 + _tmp147 * _tmp96
        _tmp149 = _tmp104 * _tmp146 + _tmp107 * _tmp144 + _tmp110 * _tmp147
        _tmp150 = _tmp114 * _tmp144 + _tmp116 * _tmp146 + _tmp119 * _tmp147
        _tmp151 = _tmp122 * _tmp144 + _tmp123 * _tmp146 + _tmp124 * _tmp147
        _tmp152 = _tmp126 * _tmp144 + _tmp128 * _tmp146 + _tmp130 * _tmp147
        _tmp153 = _tmp132 * _tmp144 + _tmp134 * _tmp146 + _tmp136 * _tmp147
        _tmp154 = _tmp138 * _tmp144 + _tmp139 * _tmp146 + _tmp140 * _tmp147
        _tmp155 = _tmp145 + _tmp98
        _tmp156 = -_tmp142 + _tmp143
        _tmp157 = _tmp83 + _tmp84
        _tmp158 = _tmp155 * _tmp82 + _tmp156 * _tmp89 + _tmp157 * _tmp96
        _tmp159 = _tmp114 * _tmp155 + _tmp116 * _tmp156 + _tmp119 * _tmp157
        _tmp160 = _tmp104 * _tmp156 + _tmp107 * _tmp155 + _tmp110 * _tmp157
        _tmp161 = _tmp122 * _tmp155 + _tmp123 * _tmp156 + _tmp124 * _tmp157
        _tmp162 = _tmp126 * _tmp155 + _tmp128 * _tmp156 + _tmp130 * _tmp157
        _tmp163 = _tmp132 * _tmp155 + _tmp134 * _tmp156 + _tmp136 * _tmp157
        _tmp164 = _tmp138 * _tmp155 + _tmp139 * _tmp156 + _tmp140 * _tmp157
        _tmp165 = _tmp0 * _tmp4
        _tmp166 = _tmp2 * _tmp80
        _tmp167 = _tmp1 * _tmp10
        _tmp168 = _tmp3 * _tmp7
        _tmp169 = -_tmp165 - _tmp166 + _tmp167 - _tmp168
        _tmp170 = (1.0 / 2.0) * _b[0]
        _tmp171 = -_tmp169 * _tmp170
        _tmp172 = _tmp0 * _tmp80
        _tmp173 = _tmp1 * _tmp7
        _tmp174 = _tmp2 * _tmp4
        _tmp175 = _tmp10 * _tmp3
        _tmp176 = (
            -1.0 / 2.0 * _tmp172
            + (1.0 / 2.0) * _tmp173
            + (1.0 / 2.0) * _tmp174
            + (1.0 / 2.0) * _tmp175
        )
        _tmp177 = -_tmp0 * _tmp7 - _tmp1 * _tmp80 - _tmp10 * _tmp2 + _tmp3 * _tmp4
        _tmp178 = (1.0 / 2.0) * _tmp177
        _tmp179 = _b[3] * _tmp178
        _tmp180 = _tmp0 * _tmp10
        _tmp181 = _tmp1 * _tmp4
        _tmp182 = _tmp2 * _tmp7
        _tmp183 = _tmp3 * _tmp80
        _tmp184 = -_tmp180 - _tmp181 + _tmp182 - _tmp183
        _tmp185 = (1.0 / 2.0) * _b[1]
        _tmp186 = _tmp179 - _tmp184 * _tmp185
        _tmp187 = _tmp170 * _tmp184
        _tmp188 = _b[2] * _tmp178
        _tmp189 = (1.0 / 2.0) * _tmp169
        _tmp190 = _b[2] * _tmp189
        _tmp191 = (1.0 / 2.0) * _b[3]
        _tmp192 = _tmp177 * _tmp185
        _tmp193 = _tmp165 + _tmp166 - _tmp167 + _tmp168
        _tmp194 = (
            (1.0 / 2.0) * _tmp172 - 1.0 / 2.0 * _tmp173 - 1.0 / 2.0 * _tmp174 - 1.0 / 2.0 * _tmp175
        )
        _tmp195 = -_b[2] * _tmp194
        _tmp196 = (1.0 / 2.0) * _b[2]
        _tmp197 = _tmp170 * _tmp177
        _tmp198 = _b[1] * _tmp194
        _tmp199 = _tmp180 + _tmp181 - _tmp182 + _tmp183
        _tmp200 = _b[3] * _tmp39
        _tmp201 = _b[0] * _tmp55
        _tmp202 = _tmp200 + _tmp201
        _tmp203 = _tmp13 * _tmp99 + _tmp24 * _tmp92 + _tmp29 * _tmp85
        _tmp204 = _b[1] * _tmp62
        _tmp205 = _b[0] * _tmp39
        _tmp206 = -_tmp204 + _tmp205
        _tmp207 = _tmp21 * _tmp92 + _tmp27 * _tmp85 + _tmp9 * _tmp99
        _tmp208 = -2 * _b[1] ** 2
        _tmp209 = -2 * _b[2] ** 2 + 1
        _tmp210 = _tmp208 + _tmp209
        _tmp211 = _tmp18 * _tmp99 + _tmp26 * _tmp92 + _tmp30 * _tmp85
        _tmp212 = _b[0] * _tmp62
        _tmp213 = _b[1] * _tmp39
        _tmp214 = _tmp212 + _tmp213
        _tmp215 = -2 * _b[0] ** 2
        _tmp216 = _tmp209 + _tmp215
        _tmp217 = -_tmp200 + _tmp201
        _tmp218 = _tmp208 + _tmp215 + 1
        _tmp219 = -_tmp212 + _tmp213
        _tmp220 = _tmp204 + _tmp205
        _tmp221 = _tmp144 * _tmp27 + _tmp146 * _tmp21 + _tmp147 * _tmp9
        _tmp222 = _tmp13 * _tmp147 + _tmp144 * _tmp29 + _tmp146 * _tmp24
        _tmp223 = _tmp144 * _tmp30 + _tmp146 * _tmp26 + _tmp147 * _tmp18
        _tmp224 = _tmp155 * _tmp27 + _tmp156 * _tmp21 + _tmp157 * _tmp9
        _tmp225 = _tmp13 * _tmp157 + _tmp155 * _tmp29 + _tmp156 * _tmp24
        _tmp226 = _tmp155 * _tmp30 + _tmp156 * _tmp26 + _tmp157 * _tmp18

        # Output terms
        _res = [0.0] * 7
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res[2] = _tmp2
        _res[3] = _tmp3
        _res[4] = (
            -_a[4] * _tmp18
            - _a[5] * _tmp13
            - _a[6] * _tmp9
            + _b[4] * _tmp18
            + _b[5] * _tmp13
            + _b[6] * _tmp9
        )
        _res[5] = (
            -_a[4] * _tmp26
            - _a[5] * _tmp24
            - _a[6] * _tmp21
            + _b[4] * _tmp26
            + _b[5] * _tmp24
            + _b[6] * _tmp21
        )
        _res[6] = (
            -_a[4] * _tmp30
            - _a[5] * _tmp29
            - _a[6] * _tmp27
            + _b[4] * _tmp30
            + _b[5] * _tmp29
            + _b[6] * _tmp27
        )
        _res_D_a = numpy.zeros((6, 6))
        _res_D_a[0, 0] = _tmp41 * _tmp42 - _tmp49 * _tmp50 - _tmp57 * _tmp58 + _tmp65 * _tmp66
        _res_D_a[0, 1] = _tmp41 * _tmp66 - _tmp42 * _tmp65 - _tmp49 * _tmp58 + _tmp50 * _tmp57
        _res_D_a[0, 2] = -_tmp41 * _tmp50 - _tmp42 * _tmp49 + _tmp57 * _tmp66 + _tmp58 * _tmp65
        _res_D_a[0, 3] = 0
        _res_D_a[0, 4] = 0
        _res_D_a[0, 5] = 0
        _res_D_a[1, 0] = _tmp42 * _tmp71 - _tmp50 * _tmp68 - _tmp58 * _tmp70 + _tmp66 * _tmp73
        _res_D_a[1, 1] = -_tmp42 * _tmp73 + _tmp50 * _tmp70 - _tmp58 * _tmp68 + _tmp66 * _tmp71
        _res_D_a[1, 2] = -_tmp42 * _tmp68 - _tmp50 * _tmp71 + _tmp58 * _tmp73 + _tmp66 * _tmp70
        _res_D_a[1, 3] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 0] = _tmp42 * _tmp76 - _tmp50 * _tmp75 - _tmp58 * _tmp74 + _tmp66 * _tmp77
        _res_D_a[2, 1] = -_tmp42 * _tmp77 + _tmp50 * _tmp74 - _tmp58 * _tmp75 + _tmp66 * _tmp76
        _res_D_a[2, 2] = -_tmp42 * _tmp75 - _tmp50 * _tmp76 + _tmp58 * _tmp77 + _tmp66 * _tmp74
        _res_D_a[2, 3] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 0] = _a[3] * _tmp125 - _tmp100 * _tmp50 - _tmp111 * _tmp58 + _tmp120 * _tmp42
        _res_D_a[3, 1] = -_a[2] * _tmp125 - _tmp100 * _tmp58 + _tmp111 * _tmp50 + _tmp120 * _tmp66
        _res_D_a[3, 2] = _a[1] * _tmp125 - _tmp100 * _tmp42 + _tmp111 * _tmp66 - _tmp120 * _tmp50
        _res_D_a[3, 3] = _tmp13 * _tmp137 + _tmp131 * _tmp9 + _tmp141 * _tmp18
        _res_D_a[3, 4] = _tmp131 * _tmp21 + _tmp137 * _tmp24 + _tmp141 * _tmp26
        _res_D_a[3, 5] = _tmp131 * _tmp27 + _tmp137 * _tmp29 + _tmp141 * _tmp30
        _res_D_a[4, 0] = -_tmp148 * _tmp50 - _tmp149 * _tmp58 + _tmp150 * _tmp42 + _tmp151 * _tmp66
        _res_D_a[4, 1] = -_tmp148 * _tmp58 + _tmp149 * _tmp50 + _tmp150 * _tmp66 - _tmp151 * _tmp42
        _res_D_a[4, 2] = -_tmp148 * _tmp42 + _tmp149 * _tmp66 - _tmp150 * _tmp50 + _tmp151 * _tmp58
        _res_D_a[4, 3] = _tmp13 * _tmp153 + _tmp152 * _tmp9 + _tmp154 * _tmp18
        _res_D_a[4, 4] = _tmp152 * _tmp21 + _tmp153 * _tmp24 + _tmp154 * _tmp26
        _res_D_a[4, 5] = _tmp152 * _tmp27 + _tmp153 * _tmp29 + _tmp154 * _tmp30
        _res_D_a[5, 0] = -_tmp158 * _tmp50 + _tmp159 * _tmp42 - _tmp160 * _tmp58 + _tmp161 * _tmp66
        _res_D_a[5, 1] = -_tmp158 * _tmp58 + _tmp159 * _tmp66 + _tmp160 * _tmp50 - _tmp161 * _tmp42
        _res_D_a[5, 2] = -_tmp158 * _tmp42 - _tmp159 * _tmp50 + _tmp160 * _tmp66 + _tmp161 * _tmp58
        _res_D_a[5, 3] = _tmp13 * _tmp163 + _tmp162 * _tmp9 + _tmp164 * _tmp18
        _res_D_a[5, 4] = _tmp162 * _tmp21 + _tmp163 * _tmp24 + _tmp164 * _tmp26
        _res_D_a[5, 5] = _tmp162 * _tmp27 + _tmp163 * _tmp29 + _tmp164 * _tmp30
        _res_D_b = numpy.zeros((6, 6))
        _res_D_b[0, 0] = _b[2] * _tmp176 + _tmp171 + _tmp186
        _res_D_b[0, 1] = _b[3] * _tmp176 - _tmp169 * _tmp185 + _tmp187 - _tmp188
        _res_D_b[0, 2] = -_b[0] * _tmp176 + _tmp184 * _tmp191 - _tmp190 + _tmp192
        _res_D_b[0, 3] = 0
        _res_D_b[0, 4] = 0
        _res_D_b[0, 5] = 0
        _res_D_b[1, 0] = _b[3] * _tmp194 - _tmp185 * _tmp193 - _tmp187 + _tmp188
        _res_D_b[1, 1] = _tmp170 * _tmp193 + _tmp186 + _tmp195
        _res_D_b[1, 2] = -_tmp184 * _tmp196 + _tmp191 * _tmp193 - _tmp197 + _tmp198
        _res_D_b[1, 3] = 0
        _res_D_b[1, 4] = 0
        _res_D_b[1, 5] = 0
        _res_D_b[2, 0] = -_b[0] * _tmp194 + _tmp190 + _tmp191 * _tmp199 - _tmp192
        _res_D_b[2, 1] = _b[3] * _tmp189 - _tmp196 * _tmp199 + _tmp197 - _tmp198
        _res_D_b[2, 2] = _tmp171 + _tmp179 + _tmp185 * _tmp199 + _tmp195
        _res_D_b[2, 3] = 0
        _res_D_b[2, 4] = 0
        _res_D_b[2, 5] = 0
        _res_D_b[3, 0] = 0
        _res_D_b[3, 1] = 0
        _res_D_b[3, 2] = 0
        _res_D_b[3, 3] = _tmp202 * _tmp203 + _tmp206 * _tmp207 + _tmp210 * _tmp211
        _res_D_b[3, 4] = _tmp203 * _tmp216 + _tmp207 * _tmp214 + _tmp211 * _tmp217
        _res_D_b[3, 5] = _tmp203 * _tmp219 + _tmp207 * _tmp218 + _tmp211 * _tmp220
        _res_D_b[4, 0] = 0
        _res_D_b[4, 1] = 0
        _res_D_b[4, 2] = 0
        _res_D_b[4, 3] = _tmp202 * _tmp222 + _tmp206 * _tmp221 + _tmp210 * _tmp223
        _res_D_b[4, 4] = _tmp214 * _tmp221 + _tmp216 * _tmp222 + _tmp217 * _tmp223
        _res_D_b[4, 5] = _tmp218 * _tmp221 + _tmp219 * _tmp222 + _tmp220 * _tmp223
        _res_D_b[5, 0] = 0
        _res_D_b[5, 1] = 0
        _res_D_b[5, 2] = 0
        _res_D_b[5, 3] = _tmp202 * _tmp225 + _tmp206 * _tmp224 + _tmp210 * _tmp226
        _res_D_b[5, 4] = _tmp214 * _tmp224 + _tmp216 * _tmp225 + _tmp217 * _tmp226
        _res_D_b[5, 5] = _tmp218 * _tmp224 + _tmp219 * _tmp225 + _tmp220 * _tmp226
        return _res, _res_D_a, _res_D_b
