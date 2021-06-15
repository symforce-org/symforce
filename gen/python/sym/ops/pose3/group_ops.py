import math
import numpy
import typing as T

import sym  # pylint: disable=unused-import


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.pose3.Pose3'>.
    """

    @staticmethod
    def identity():
        # type: () -> T.List[float]

        # Total ops: 0

        # Input arrays

        # Intermediate terms (0)

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
        # type: (sym.Pose3) -> T.List[float]

        # Total ops: 50

        # Input arrays
        _a = a.data

        # Intermediate terms (11)
        _tmp0 = -2 * _a[1] ** 2
        _tmp1 = 1 - 2 * _a[2] ** 2
        _tmp2 = 2 * _a[0]
        _tmp3 = _a[1] * _tmp2
        _tmp4 = 2 * _a[2]
        _tmp5 = _a[3] * _tmp4
        _tmp6 = 2 * _a[1] * _a[3]
        _tmp7 = _a[2] * _tmp2
        _tmp8 = -2 * _a[0] ** 2
        _tmp9 = _a[3] * _tmp2
        _tmp10 = _a[1] * _tmp4

        # Output terms
        _res = [0.0] * 7
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        _res[4] = -_a[4] * (_tmp0 + _tmp1) - _a[5] * (_tmp3 + _tmp5) - _a[6] * (-_tmp6 + _tmp7)
        _res[5] = -_a[4] * (_tmp3 - _tmp5) - _a[5] * (_tmp1 + _tmp8) - _a[6] * (_tmp10 + _tmp9)
        _res[6] = -_a[4] * (_tmp6 + _tmp7) - _a[5] * (_tmp10 - _tmp9) - _a[6] * (_tmp0 + _tmp8 + 1)
        return _res

    @staticmethod
    def compose(a, b):
        # type: (sym.Pose3, sym.Pose3) -> T.List[float]

        # Total ops: 79

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (11)
        _tmp0 = -2 * _a[1] ** 2
        _tmp1 = -2 * _a[2] ** 2
        _tmp2 = 2 * _a[0] * _a[1]
        _tmp3 = 2 * _a[3]
        _tmp4 = _a[2] * _tmp3
        _tmp5 = _a[1] * _tmp3
        _tmp6 = 2 * _a[2]
        _tmp7 = _a[0] * _tmp6
        _tmp8 = 1 - 2 * _a[0] ** 2
        _tmp9 = _a[0] * _tmp3
        _tmp10 = _a[1] * _tmp6

        # Output terms
        _res = [0.0] * 7
        _res[0] = _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0]
        _res[1] = -_a[0] * _b[2] + _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1]
        _res[2] = _a[0] * _b[1] - _a[1] * _b[0] + _a[2] * _b[3] + _a[3] * _b[2]
        _res[3] = -_a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] + _a[3] * _b[3]
        _res[4] = (
            _a[4] + _b[4] * (_tmp0 + _tmp1 + 1) + _b[5] * (_tmp2 - _tmp4) + _b[6] * (_tmp5 + _tmp7)
        )
        _res[5] = (
            _a[5] + _b[4] * (_tmp2 + _tmp4) + _b[5] * (_tmp1 + _tmp8) + _b[6] * (_tmp10 - _tmp9)
        )
        _res[6] = (
            _a[6] + _b[4] * (-_tmp5 + _tmp7) + _b[5] * (_tmp10 + _tmp9) + _b[6] * (_tmp0 + _tmp8)
        )
        return _res

    @staticmethod
    def between(a, b):
        # type: (sym.Pose3, sym.Pose3) -> T.List[float]

        # Total ops: 103

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (20)
        _tmp0 = -2 * _a[1] ** 2
        _tmp1 = -2 * _a[2] ** 2
        _tmp2 = _tmp0 + _tmp1 + 1
        _tmp3 = 2 * _a[0]
        _tmp4 = _a[1] * _tmp3
        _tmp5 = 2 * _a[2]
        _tmp6 = _a[3] * _tmp5
        _tmp7 = _tmp4 + _tmp6
        _tmp8 = 2 * _a[1] * _a[3]
        _tmp9 = _a[2] * _tmp3
        _tmp10 = -_tmp8 + _tmp9
        _tmp11 = 1 - 2 * _a[0] ** 2
        _tmp12 = _tmp1 + _tmp11
        _tmp13 = _tmp4 - _tmp6
        _tmp14 = _a[3] * _tmp3
        _tmp15 = _a[1] * _tmp5
        _tmp16 = _tmp14 + _tmp15
        _tmp17 = _tmp0 + _tmp11
        _tmp18 = -_tmp14 + _tmp15
        _tmp19 = _tmp8 + _tmp9

        # Output terms
        _res = [0.0] * 7
        _res[0] = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0]
        _res[1] = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1]
        _res[2] = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2]
        _res[3] = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3]
        _res[4] = (
            -_a[4] * _tmp2
            - _a[5] * _tmp7
            - _a[6] * _tmp10
            + _b[4] * _tmp2
            + _b[5] * _tmp7
            + _b[6] * _tmp10
        )
        _res[5] = (
            -_a[4] * _tmp13
            - _a[5] * _tmp12
            - _a[6] * _tmp16
            + _b[4] * _tmp13
            + _b[5] * _tmp12
            + _b[6] * _tmp16
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
        # type: (sym.Pose3) -> T.Tuple[T.List[float], numpy.ndarray]

        # Total ops: 222

        # Input arrays
        _a = a.data

        # Intermediate terms (67)
        _tmp0 = _a[1] ** 2
        _tmp1 = -2 * _tmp0
        _tmp2 = _a[2] ** 2
        _tmp3 = -2 * _tmp2
        _tmp4 = _tmp1 + _tmp3 + 1
        _tmp5 = 2 * _a[0]
        _tmp6 = _a[1] * _tmp5
        _tmp7 = 2 * _a[2]
        _tmp8 = _a[3] * _tmp7
        _tmp9 = _tmp6 + _tmp8
        _tmp10 = 2 * _a[1]
        _tmp11 = _a[3] * _tmp10
        _tmp12 = -_tmp11
        _tmp13 = _a[0] * _tmp7
        _tmp14 = _tmp12 + _tmp13
        _tmp15 = _a[0] ** 2
        _tmp16 = 1 - 2 * _tmp15
        _tmp17 = _tmp16 + _tmp3
        _tmp18 = -_tmp8
        _tmp19 = _tmp18 + _tmp6
        _tmp20 = _a[3] * _tmp5
        _tmp21 = _a[1] * _tmp7
        _tmp22 = _tmp20 + _tmp21
        _tmp23 = _tmp1 + _tmp16
        _tmp24 = -_tmp20
        _tmp25 = _tmp21 + _tmp24
        _tmp26 = _tmp11 + _tmp13
        _tmp27 = -_a[3] ** 2
        _tmp28 = -_tmp6
        _tmp29 = -_tmp13
        _tmp30 = _tmp15 + _tmp27
        _tmp31 = -_tmp21
        _tmp32 = _a[5] * _tmp10
        _tmp33 = _a[6] * _tmp7
        _tmp34 = _tmp32 + _tmp33
        _tmp35 = (1.0 / 2.0) * _a[3]
        _tmp36 = _a[6] * _tmp10
        _tmp37 = _a[5] * _tmp7
        _tmp38 = -_tmp36 + _tmp37
        _tmp39 = (1.0 / 2.0) * _a[0]
        _tmp40 = _a[6] * _tmp5
        _tmp41 = 2 * _a[3]
        _tmp42 = _a[5] * _tmp41
        _tmp43 = 4 * _a[4]
        _tmp44 = -_a[2] * _tmp43 + _tmp40 + _tmp42
        _tmp45 = (1.0 / 2.0) * _a[1]
        _tmp46 = _a[5] * _tmp5
        _tmp47 = _a[6] * _tmp41
        _tmp48 = -_a[1] * _tmp43 + _tmp46 - _tmp47
        _tmp49 = (1.0 / 2.0) * _a[2]
        _tmp50 = (1.0 / 2.0) * _tmp44
        _tmp51 = _a[4] * _tmp7
        _tmp52 = _tmp40 - _tmp51
        _tmp53 = _a[4] * _tmp5
        _tmp54 = _tmp33 + _tmp53
        _tmp55 = _a[4] * _tmp41
        _tmp56 = 4 * _a[5]
        _tmp57 = -_a[2] * _tmp56 + _tmp36 - _tmp55
        _tmp58 = _a[4] * _tmp10
        _tmp59 = -_a[0] * _tmp56 + _tmp47 + _tmp58
        _tmp60 = (1.0 / 2.0) * _tmp54
        _tmp61 = -_tmp46 + _tmp58
        _tmp62 = (1.0 / 2.0) * _tmp61
        _tmp63 = (1.0 / 2.0) * _tmp32 + (1.0 / 2.0) * _tmp53
        _tmp64 = 4 * _a[6]
        _tmp65 = -_a[0] * _tmp64 - _tmp42 + _tmp51
        _tmp66 = -_a[1] * _tmp64 + _tmp37 + _tmp55

        # Output terms
        _res = [0.0] * 7
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        _res[4] = -_a[4] * _tmp4 - _a[5] * _tmp9 - _a[6] * _tmp14
        _res[5] = -_a[4] * _tmp19 - _a[5] * _tmp17 - _a[6] * _tmp22
        _res[6] = -_a[4] * _tmp26 - _a[5] * _tmp25 - _a[6] * _tmp23
        _res_D_a = numpy.zeros((6, 6))
        _res_D_a[0, 0] = _tmp0 - _tmp15 + _tmp2 + _tmp27
        _res_D_a[0, 1] = _tmp28 + _tmp8
        _res_D_a[0, 2] = _tmp12 + _tmp29
        _res_D_a[0, 3] = 0
        _res_D_a[0, 4] = 0
        _res_D_a[0, 5] = 0
        _res_D_a[1, 0] = _tmp18 + _tmp28
        _res_D_a[1, 1] = -_tmp0 + _tmp2 + _tmp30
        _res_D_a[1, 2] = _tmp20 + _tmp31
        _res_D_a[1, 3] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 0] = _tmp11 + _tmp29
        _res_D_a[2, 1] = _tmp24 + _tmp31
        _res_D_a[2, 2] = _tmp0 - _tmp2 + _tmp30
        _res_D_a[2, 3] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 0] = -_tmp34 * _tmp35 + _tmp38 * _tmp39 + _tmp44 * _tmp45 - _tmp48 * _tmp49
        _res_D_a[3, 1] = -_a[0] * _tmp50 + _tmp34 * _tmp49 - _tmp35 * _tmp48 + _tmp38 * _tmp45
        _res_D_a[3, 2] = -_a[3] * _tmp50 - _tmp34 * _tmp45 + _tmp38 * _tmp49 + _tmp39 * _tmp48
        _res_D_a[3, 3] = -_tmp4
        _res_D_a[3, 4] = -_tmp9
        _res_D_a[3, 5] = -_tmp14
        _res_D_a[4, 0] = -_tmp35 * _tmp59 + _tmp39 * _tmp52 + _tmp45 * _tmp57 - _tmp49 * _tmp54
        _res_D_a[4, 1] = -_a[3] * _tmp60 - _tmp39 * _tmp57 + _tmp45 * _tmp52 + _tmp49 * _tmp59
        _res_D_a[4, 2] = _a[0] * _tmp60 - _tmp35 * _tmp57 - _tmp45 * _tmp59 + _tmp49 * _tmp52
        _res_D_a[4, 3] = -_tmp19
        _res_D_a[4, 4] = -_tmp17
        _res_D_a[4, 5] = -_tmp22
        _res_D_a[5, 0] = _a[0] * _tmp62 + _a[1] * _tmp63 - _tmp35 * _tmp65 - _tmp49 * _tmp66
        _res_D_a[5, 1] = -_a[0] * _tmp63 + _a[1] * _tmp62 - _tmp35 * _tmp66 + _tmp49 * _tmp65
        _res_D_a[5, 2] = -_a[3] * _tmp63 + _tmp39 * _tmp66 - _tmp45 * _tmp65 + _tmp49 * _tmp61
        _res_D_a[5, 3] = -_tmp26
        _res_D_a[5, 4] = -_tmp25
        _res_D_a[5, 5] = -_tmp23
        return _res, _res_D_a

    @staticmethod
    def compose_with_jacobians(a, b):
        # type: (sym.Pose3, sym.Pose3) -> T.Tuple[T.List[float], numpy.ndarray, numpy.ndarray]

        # Total ops: 484

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (133)
        _tmp0 = _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0]
        _tmp1 = -_a[0] * _b[2] + _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1]
        _tmp2 = _a[0] * _b[1] - _a[1] * _b[0] + _a[2] * _b[3] + _a[3] * _b[2]
        _tmp3 = -_a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] + _a[3] * _b[3]
        _tmp4 = -2 * _a[1] ** 2
        _tmp5 = 1 - 2 * _a[2] ** 2
        _tmp6 = _tmp4 + _tmp5
        _tmp7 = 2 * _a[0]
        _tmp8 = _a[1] * _tmp7
        _tmp9 = 2 * _a[3]
        _tmp10 = _a[2] * _tmp9
        _tmp11 = -_tmp10 + _tmp8
        _tmp12 = _a[1] * _tmp9
        _tmp13 = 2 * _a[2]
        _tmp14 = _a[0] * _tmp13
        _tmp15 = _tmp12 + _tmp14
        _tmp16 = -2 * _a[0] ** 2
        _tmp17 = _tmp16 + _tmp5
        _tmp18 = _tmp10 + _tmp8
        _tmp19 = _a[0] * _tmp9
        _tmp20 = _a[1] * _tmp13
        _tmp21 = -_tmp19 + _tmp20
        _tmp22 = _tmp16 + _tmp4 + 1
        _tmp23 = _tmp19 + _tmp20
        _tmp24 = -_tmp12 + _tmp14
        _tmp25 = 2 * _tmp2
        _tmp26 = _b[0] * _tmp25
        _tmp27 = 2 * _tmp3
        _tmp28 = _b[1] * _tmp27
        _tmp29 = 2 * _tmp1
        _tmp30 = _b[3] * _tmp29
        _tmp31 = 2 * _tmp0
        _tmp32 = _b[2] * _tmp31
        _tmp33 = -_tmp30 + _tmp32
        _tmp34 = _tmp26 - _tmp28 + _tmp33
        _tmp35 = (1.0 / 2.0) * _a[1]
        _tmp36 = _b[2] * _tmp29
        _tmp37 = _b[3] * _tmp31
        _tmp38 = -_tmp37
        _tmp39 = _b[1] * _tmp25
        _tmp40 = _b[0] * _tmp27
        _tmp41 = _tmp39 + _tmp40
        _tmp42 = -_tmp36 + _tmp38 + _tmp41
        _tmp43 = (1.0 / 2.0) * _a[0]
        _tmp44 = _b[1] * _tmp29
        _tmp45 = -_tmp44
        _tmp46 = _b[0] * _tmp31
        _tmp47 = _b[2] * _tmp25
        _tmp48 = _b[3] * _tmp27
        _tmp49 = -_tmp47 + _tmp48
        _tmp50 = _tmp45 + _tmp46 + _tmp49
        _tmp51 = (1.0 / 2.0) * _a[3]
        _tmp52 = _b[3] * _tmp25
        _tmp53 = _b[2] * _tmp27
        _tmp54 = _b[0] * _tmp29
        _tmp55 = _b[1] * _tmp31
        _tmp56 = _tmp54 + _tmp55
        _tmp57 = _tmp52 + _tmp53 + _tmp56
        _tmp58 = (1.0 / 2.0) * _a[2]
        _tmp59 = -_tmp26 + _tmp28 + _tmp33
        _tmp60 = _tmp36 + _tmp37 + _tmp41
        _tmp61 = -_tmp46
        _tmp62 = _tmp44 + _tmp49 + _tmp61
        _tmp63 = -_tmp52
        _tmp64 = -_tmp53 + _tmp56 + _tmp63
        _tmp65 = _tmp26 + _tmp28 + _tmp30 + _tmp32
        _tmp66 = _tmp36 + _tmp38 + _tmp39 - _tmp40
        _tmp67 = _tmp45 + _tmp47 + _tmp48 + _tmp61
        _tmp68 = _tmp53 + _tmp54 - _tmp55 + _tmp63
        _tmp69 = 2 * _a[1]
        _tmp70 = _b[6] * _tmp69
        _tmp71 = _b[5] * _tmp13
        _tmp72 = _tmp70 - _tmp71
        _tmp73 = _b[5] * _tmp69
        _tmp74 = _b[6] * _tmp13
        _tmp75 = _tmp73 + _tmp74
        _tmp76 = _a[0] * _b[6]
        _tmp77 = 2 * _tmp76
        _tmp78 = _b[5] * _tmp9
        _tmp79 = -4 * _a[2] * _b[4] + _tmp77 - _tmp78
        _tmp80 = 4 * _a[1]
        _tmp81 = _b[5] * _tmp7
        _tmp82 = _b[6] * _tmp9
        _tmp83 = -_b[4] * _tmp80 + _tmp81 + _tmp82
        _tmp84 = _b[4] * _tmp13
        _tmp85 = -_tmp77 + _tmp84
        _tmp86 = _b[4] * _tmp7
        _tmp87 = _tmp74 + _tmp86
        _tmp88 = _b[4] * _tmp9
        _tmp89 = 4 * _b[5]
        _tmp90 = -_a[2] * _tmp89 + _tmp70 + _tmp88
        _tmp91 = _b[4] * _tmp69
        _tmp92 = -_a[0] * _tmp89 - _tmp82 + _tmp91
        _tmp93 = _tmp81 - _tmp91
        _tmp94 = (1.0 / 2.0) * _tmp93
        _tmp95 = _tmp73 + _tmp86
        _tmp96 = -_b[6] * _tmp80 + _tmp71 - _tmp88
        _tmp97 = -4 * _tmp76 + _tmp78 + _tmp84
        _tmp98 = _a[0] * _tmp25
        _tmp99 = _tmp1 * _tmp9
        _tmp100 = _a[2] * _tmp31
        _tmp101 = _a[1] * _tmp27
        _tmp102 = _tmp100 + _tmp101 - _tmp98 - _tmp99
        _tmp103 = (1.0 / 2.0) * _b[1]
        _tmp104 = -_tmp102 * _tmp103
        _tmp105 = _tmp2 * _tmp9
        _tmp106 = _a[0] * _tmp29
        _tmp107 = _a[1] * _tmp31
        _tmp108 = _a[2] * _tmp27
        _tmp109 = _tmp105 - _tmp106 + _tmp107 - _tmp108
        _tmp110 = (1.0 / 2.0) * _b[2]
        _tmp111 = _a[0] * _tmp31 + _a[1] * _tmp29 + _a[2] * _tmp25 + _a[3] * _tmp27
        _tmp112 = (1.0 / 2.0) * _b[3]
        _tmp113 = _tmp111 * _tmp112
        _tmp114 = _a[1] * _tmp25
        _tmp115 = _a[2] * _tmp29
        _tmp116 = _a[3] * _tmp31
        _tmp117 = _a[0] * _tmp27
        _tmp118 = (
            (1.0 / 2.0) * _tmp114
            - 1.0 / 2.0 * _tmp115
            - 1.0 / 2.0 * _tmp116
            + (1.0 / 2.0) * _tmp117
        )
        _tmp119 = -_b[0] * _tmp118 + _tmp113
        _tmp120 = _tmp110 * _tmp111
        _tmp121 = (1.0 / 2.0) * _b[0]
        _tmp122 = _tmp102 * _tmp121
        _tmp123 = (1.0 / 2.0) * _tmp109
        _tmp124 = _tmp103 * _tmp111
        _tmp125 = _b[2] * _tmp118
        _tmp126 = (
            -1.0 / 2.0 * _tmp114
            + (1.0 / 2.0) * _tmp115
            + (1.0 / 2.0) * _tmp116
            - 1.0 / 2.0 * _tmp117
        )
        _tmp127 = -_tmp105 + _tmp106 - _tmp107 + _tmp108
        _tmp128 = (1.0 / 2.0) * _tmp127
        _tmp129 = -_tmp110 * _tmp127
        _tmp130 = _tmp111 * _tmp121
        _tmp131 = _b[1] * _tmp128
        _tmp132 = -_tmp100 - _tmp101 + _tmp98 + _tmp99

        # Output terms
        _res = [0.0] * 7
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res[2] = _tmp2
        _res[3] = _tmp3
        _res[4] = _a[4] + _b[4] * _tmp6 + _b[5] * _tmp11 + _b[6] * _tmp15
        _res[5] = _a[5] + _b[4] * _tmp18 + _b[5] * _tmp17 + _b[6] * _tmp21
        _res[6] = _a[6] + _b[4] * _tmp24 + _b[5] * _tmp23 + _b[6] * _tmp22
        _res_D_a = numpy.zeros((6, 6))
        _res_D_a[0, 0] = -_tmp34 * _tmp35 - _tmp42 * _tmp43 + _tmp50 * _tmp51 + _tmp57 * _tmp58
        _res_D_a[0, 1] = _tmp34 * _tmp43 - _tmp35 * _tmp42 - _tmp50 * _tmp58 + _tmp51 * _tmp57
        _res_D_a[0, 2] = _tmp34 * _tmp51 + _tmp35 * _tmp50 - _tmp42 * _tmp58 - _tmp43 * _tmp57
        _res_D_a[0, 3] = 0
        _res_D_a[0, 4] = 0
        _res_D_a[0, 5] = 0
        _res_D_a[1, 0] = -_tmp35 * _tmp60 - _tmp43 * _tmp59 + _tmp51 * _tmp64 + _tmp58 * _tmp62
        _res_D_a[1, 1] = -_tmp35 * _tmp59 + _tmp43 * _tmp60 + _tmp51 * _tmp62 - _tmp58 * _tmp64
        _res_D_a[1, 2] = _tmp35 * _tmp64 - _tmp43 * _tmp62 + _tmp51 * _tmp60 - _tmp58 * _tmp59
        _res_D_a[1, 3] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 0] = -_tmp35 * _tmp67 - _tmp43 * _tmp68 + _tmp51 * _tmp65 + _tmp58 * _tmp66
        _res_D_a[2, 1] = -_tmp35 * _tmp68 + _tmp43 * _tmp67 + _tmp51 * _tmp66 - _tmp58 * _tmp65
        _res_D_a[2, 2] = _tmp35 * _tmp65 - _tmp43 * _tmp66 + _tmp51 * _tmp67 - _tmp58 * _tmp68
        _res_D_a[2, 3] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 0] = -_tmp35 * _tmp79 - _tmp43 * _tmp72 + _tmp51 * _tmp75 + _tmp58 * _tmp83
        _res_D_a[3, 1] = -_tmp35 * _tmp72 + _tmp43 * _tmp79 + _tmp51 * _tmp83 - _tmp58 * _tmp75
        _res_D_a[3, 2] = _tmp35 * _tmp75 - _tmp43 * _tmp83 + _tmp51 * _tmp79 - _tmp58 * _tmp72
        _res_D_a[3, 3] = 1
        _res_D_a[3, 4] = 0
        _res_D_a[3, 5] = 0
        _res_D_a[4, 0] = -_tmp35 * _tmp90 - _tmp43 * _tmp85 + _tmp51 * _tmp92 + _tmp58 * _tmp87
        _res_D_a[4, 1] = -_tmp35 * _tmp85 + _tmp43 * _tmp90 + _tmp51 * _tmp87 - _tmp58 * _tmp92
        _res_D_a[4, 2] = _tmp35 * _tmp92 - _tmp43 * _tmp87 + _tmp51 * _tmp90 - _tmp58 * _tmp85
        _res_D_a[4, 3] = 0
        _res_D_a[4, 4] = 1
        _res_D_a[4, 5] = 0
        _res_D_a[5, 0] = -_a[0] * _tmp94 - _tmp35 * _tmp95 + _tmp51 * _tmp97 + _tmp58 * _tmp96
        _res_D_a[5, 1] = -_tmp35 * _tmp93 + _tmp43 * _tmp95 + _tmp51 * _tmp96 - _tmp58 * _tmp97
        _res_D_a[5, 2] = -_a[2] * _tmp94 + _tmp35 * _tmp97 - _tmp43 * _tmp96 + _tmp51 * _tmp95
        _res_D_a[5, 3] = 0
        _res_D_a[5, 4] = 0
        _res_D_a[5, 5] = 1
        _res_D_b = numpy.zeros((6, 6))
        _res_D_b[0, 0] = _tmp104 + _tmp109 * _tmp110 + _tmp119
        _res_D_b[0, 1] = -_b[1] * _tmp118 + _b[3] * _tmp123 - _tmp120 + _tmp122
        _res_D_b[0, 2] = -_b[0] * _tmp123 + _tmp102 * _tmp112 + _tmp124 - _tmp125
        _res_D_b[0, 3] = 0
        _res_D_b[0, 4] = 0
        _res_D_b[0, 5] = 0
        _res_D_b[1, 0] = -_b[1] * _tmp126 + _b[3] * _tmp128 + _tmp120 - _tmp122
        _res_D_b[1, 1] = _b[0] * _tmp126 + _tmp104 + _tmp113 + _tmp129
        _res_D_b[1, 2] = _b[3] * _tmp126 - _tmp102 * _tmp110 - _tmp130 + _tmp131
        _res_D_b[1, 3] = 0
        _res_D_b[1, 4] = 0
        _res_D_b[1, 5] = 0
        _res_D_b[2, 0] = -_b[0] * _tmp128 + _tmp112 * _tmp132 - _tmp124 + _tmp125
        _res_D_b[2, 1] = _b[3] * _tmp118 - _tmp110 * _tmp132 + _tmp130 - _tmp131
        _res_D_b[2, 2] = _tmp103 * _tmp132 + _tmp119 + _tmp129
        _res_D_b[2, 3] = 0
        _res_D_b[2, 4] = 0
        _res_D_b[2, 5] = 0
        _res_D_b[3, 0] = 0
        _res_D_b[3, 1] = 0
        _res_D_b[3, 2] = 0
        _res_D_b[3, 3] = _tmp6
        _res_D_b[3, 4] = _tmp11
        _res_D_b[3, 5] = _tmp15
        _res_D_b[4, 0] = 0
        _res_D_b[4, 1] = 0
        _res_D_b[4, 2] = 0
        _res_D_b[4, 3] = _tmp18
        _res_D_b[4, 4] = _tmp17
        _res_D_b[4, 5] = _tmp21
        _res_D_b[5, 0] = 0
        _res_D_b[5, 1] = 0
        _res_D_b[5, 2] = 0
        _res_D_b[5, 3] = _tmp24
        _res_D_b[5, 4] = _tmp23
        _res_D_b[5, 5] = _tmp22
        return _res, _res_D_a, _res_D_b

    @staticmethod
    def between_with_jacobians(a, b):
        # type: (sym.Pose3, sym.Pose3) -> T.Tuple[T.List[float], numpy.ndarray, numpy.ndarray]

        # Total ops: 586

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (149)
        _tmp0 = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0]
        _tmp1 = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1]
        _tmp2 = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2]
        _tmp3 = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3]
        _tmp4 = 2 * _a[1] ** 2
        _tmp5 = -_tmp4
        _tmp6 = 2 * _a[2] ** 2
        _tmp7 = 1 - _tmp6
        _tmp8 = _tmp5 + _tmp7
        _tmp9 = 2 * _a[0]
        _tmp10 = _a[1] * _tmp9
        _tmp11 = 2 * _a[2]
        _tmp12 = _a[3] * _tmp11
        _tmp13 = _tmp10 + _tmp12
        _tmp14 = 2 * _a[3]
        _tmp15 = _a[1] * _tmp14
        _tmp16 = -_tmp15
        _tmp17 = _a[0] * _tmp11
        _tmp18 = _tmp16 + _tmp17
        _tmp19 = 2 * _a[0] ** 2
        _tmp20 = -_tmp19
        _tmp21 = _tmp20 + _tmp7
        _tmp22 = -_tmp12
        _tmp23 = _tmp10 + _tmp22
        _tmp24 = _a[0] * _tmp14
        _tmp25 = _a[1] * _tmp11
        _tmp26 = _tmp24 + _tmp25
        _tmp27 = _tmp20 + _tmp5 + 1
        _tmp28 = -_tmp24
        _tmp29 = _tmp25 + _tmp28
        _tmp30 = _tmp15 + _tmp17
        _tmp31 = 2 * _tmp1
        _tmp32 = _b[0] * _tmp31
        _tmp33 = -_tmp32
        _tmp34 = 2 * _tmp3
        _tmp35 = _b[2] * _tmp34
        _tmp36 = 2 * _tmp2
        _tmp37 = _b[3] * _tmp36
        _tmp38 = 2 * _tmp0
        _tmp39 = -_b[1] * _tmp38
        _tmp40 = -_tmp37 + _tmp39
        _tmp41 = _tmp33 - _tmp35 + _tmp40
        _tmp42 = (1.0 / 2.0) * _a[2]
        _tmp43 = _b[3] * _tmp31
        _tmp44 = _b[1] * _tmp34
        _tmp45 = -_b[0] * _tmp36
        _tmp46 = _b[2] * _tmp38
        _tmp47 = _tmp45 - _tmp46
        _tmp48 = _tmp43 + _tmp44 + _tmp47
        _tmp49 = (1.0 / 2.0) * _a[1]
        _tmp50 = _b[1] * _tmp31
        _tmp51 = _b[0] * _tmp38
        _tmp52 = _b[2] * _tmp36
        _tmp53 = -_b[3] * _tmp34
        _tmp54 = _tmp52 + _tmp53
        _tmp55 = _tmp50 - _tmp51 + _tmp54
        _tmp56 = (1.0 / 2.0) * _a[3]
        _tmp57 = _b[1] * _tmp36
        _tmp58 = _b[3] * _tmp38
        _tmp59 = -_tmp58
        _tmp60 = -_b[2] * _tmp31
        _tmp61 = _b[0] * _tmp34
        _tmp62 = _tmp60 + _tmp61
        _tmp63 = _tmp57 + _tmp59 + _tmp62
        _tmp64 = (1.0 / 2.0) * _a[0]
        _tmp65 = _tmp33 + _tmp35 + _tmp37 + _tmp39
        _tmp66 = -_tmp43
        _tmp67 = _tmp44 + _tmp45 + _tmp46 + _tmp66
        _tmp68 = -_tmp50 + _tmp51 + _tmp54
        _tmp69 = -_tmp57
        _tmp70 = _tmp59 + _tmp60 - _tmp61 + _tmp69
        _tmp71 = _tmp32 + _tmp35 + _tmp40
        _tmp72 = -_tmp44 + _tmp47 + _tmp66
        _tmp73 = _tmp50 + _tmp51 - _tmp52 + _tmp53
        _tmp74 = _tmp58 + _tmp62 + _tmp69
        _tmp75 = 2 * _a[1]
        _tmp76 = _b[6] * _tmp75
        _tmp77 = _a[6] * _tmp75
        _tmp78 = -_a[5] * _tmp11 + _b[5] * _tmp11
        _tmp79 = -_tmp76 + _tmp77 + _tmp78
        _tmp80 = 4 * _a[1]
        _tmp81 = 2 * _a[5]
        _tmp82 = _a[0] * _tmp81
        _tmp83 = _b[5] * _tmp9
        _tmp84 = _b[6] * _tmp14
        _tmp85 = _a[6] * _tmp14
        _tmp86 = (
            (1.0 / 2.0) * _a[4] * _tmp80
            - 1.0 / 2.0 * _b[4] * _tmp80
            - 1.0 / 2.0 * _tmp82
            + (1.0 / 2.0) * _tmp83
            - 1.0 / 2.0 * _tmp84
            + (1.0 / 2.0) * _tmp85
        )
        _tmp87 = -_a[1] * _tmp81 + _b[5] * _tmp75
        _tmp88 = -_a[6] * _tmp11 + _b[6] * _tmp11
        _tmp89 = _tmp87 + _tmp88
        _tmp90 = _b[5] * _tmp14
        _tmp91 = _a[5] * _tmp14
        _tmp92 = 4 * _a[2]
        _tmp93 = -_a[6] * _tmp9 + _b[6] * _tmp9
        _tmp94 = _a[4] * _tmp92 - _b[4] * _tmp92 + _tmp90 - _tmp91 + _tmp93
        _tmp95 = _tmp6 - 1
        _tmp96 = -_tmp10
        _tmp97 = -_tmp17
        _tmp98 = _b[4] * _tmp14
        _tmp99 = _a[4] * _tmp14
        _tmp100 = _a[5] * _tmp92 - _b[5] * _tmp92 + _tmp76 - _tmp77 - _tmp98 + _tmp99
        _tmp101 = 4 * _a[0]
        _tmp102 = 2 * _a[4]
        _tmp103 = -_a[1] * _tmp102 + _b[4] * _tmp75
        _tmp104 = _a[5] * _tmp101 - _b[5] * _tmp101 + _tmp103 + _tmp84 - _tmp85
        _tmp105 = -_a[0] * _tmp102 + _b[4] * _tmp9
        _tmp106 = _tmp105 + _tmp88
        _tmp107 = _b[4] * _tmp11
        _tmp108 = _a[4] * _tmp11
        _tmp109 = -_tmp107 + _tmp108 + _tmp93
        _tmp110 = -_tmp25
        _tmp111 = _a[6] * _tmp80 - _b[6] * _tmp80 + _tmp78 + _tmp98 - _tmp99
        _tmp112 = _tmp103 + _tmp82 - _tmp83
        _tmp113 = _tmp105 + _tmp87
        _tmp114 = _a[6] * _tmp101 - _b[6] * _tmp101 + _tmp107 - _tmp108 - _tmp90 + _tmp91
        _tmp115 = _a[0] * _tmp36
        _tmp116 = _a[3] * _tmp31
        _tmp117 = _a[2] * _tmp38
        _tmp118 = _a[1] * _tmp34
        _tmp119 = _tmp115 - _tmp116 - _tmp117 - _tmp118
        _tmp120 = (1.0 / 2.0) * _b[1]
        _tmp121 = -_tmp119 * _tmp120
        _tmp122 = _a[3] * _tmp36
        _tmp123 = _a[0] * _tmp31
        _tmp124 = _a[1] * _tmp38
        _tmp125 = _a[2] * _tmp34
        _tmp126 = _tmp122 + _tmp123 - _tmp124 + _tmp125
        _tmp127 = (1.0 / 2.0) * _b[2]
        _tmp128 = -_a[0] * _tmp38 - _a[1] * _tmp31 + _a[3] * _tmp34 - _tmp11 * _tmp2
        _tmp129 = (1.0 / 2.0) * _b[3]
        _tmp130 = _tmp128 * _tmp129
        _tmp131 = _a[1] * _tmp36
        _tmp132 = _tmp1 * _tmp11
        _tmp133 = _a[3] * _tmp38
        _tmp134 = _a[0] * _tmp34
        _tmp135 = -_tmp131 + _tmp132 - _tmp133 - _tmp134
        _tmp136 = (1.0 / 2.0) * _b[0]
        _tmp137 = _tmp130 - _tmp135 * _tmp136
        _tmp138 = _tmp119 * _tmp136
        _tmp139 = _tmp127 * _tmp128
        _tmp140 = _tmp120 * _tmp128
        _tmp141 = _tmp127 * _tmp135
        _tmp142 = _tmp131 - _tmp132 + _tmp133 + _tmp134
        _tmp143 = -_tmp122 - _tmp123 + _tmp124 - _tmp125
        _tmp144 = (1.0 / 2.0) * _tmp143
        _tmp145 = -_b[2] * _tmp144
        _tmp146 = _tmp128 * _tmp136
        _tmp147 = _b[1] * _tmp144
        _tmp148 = -_tmp115 + _tmp116 + _tmp117 + _tmp118

        # Output terms
        _res = [0.0] * 7
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res[2] = _tmp2
        _res[3] = _tmp3
        _res[4] = (
            -_a[4] * _tmp8
            - _a[5] * _tmp13
            - _a[6] * _tmp18
            + _b[4] * _tmp8
            + _b[5] * _tmp13
            + _b[6] * _tmp18
        )
        _res[5] = (
            -_a[4] * _tmp23
            - _a[5] * _tmp21
            - _a[6] * _tmp26
            + _b[4] * _tmp23
            + _b[5] * _tmp21
            + _b[6] * _tmp26
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
        _res_D_a[0, 0] = _tmp41 * _tmp42 - _tmp48 * _tmp49 + _tmp55 * _tmp56 - _tmp63 * _tmp64
        _res_D_a[0, 1] = _tmp41 * _tmp56 - _tmp42 * _tmp55 + _tmp48 * _tmp64 - _tmp49 * _tmp63
        _res_D_a[0, 2] = -_tmp41 * _tmp64 - _tmp42 * _tmp63 + _tmp48 * _tmp56 + _tmp49 * _tmp55
        _res_D_a[0, 3] = 0
        _res_D_a[0, 4] = 0
        _res_D_a[0, 5] = 0
        _res_D_a[1, 0] = _tmp42 * _tmp68 - _tmp49 * _tmp70 + _tmp56 * _tmp65 - _tmp64 * _tmp67
        _res_D_a[1, 1] = -_tmp42 * _tmp65 - _tmp49 * _tmp67 + _tmp56 * _tmp68 + _tmp64 * _tmp70
        _res_D_a[1, 2] = -_tmp42 * _tmp67 + _tmp49 * _tmp65 + _tmp56 * _tmp70 - _tmp64 * _tmp68
        _res_D_a[1, 3] = 0
        _res_D_a[1, 4] = 0
        _res_D_a[1, 5] = 0
        _res_D_a[2, 0] = _tmp42 * _tmp74 - _tmp49 * _tmp73 + _tmp56 * _tmp72 - _tmp64 * _tmp71
        _res_D_a[2, 1] = -_tmp42 * _tmp72 - _tmp49 * _tmp71 + _tmp56 * _tmp74 + _tmp64 * _tmp73
        _res_D_a[2, 2] = -_tmp42 * _tmp71 + _tmp49 * _tmp72 + _tmp56 * _tmp73 - _tmp64 * _tmp74
        _res_D_a[2, 3] = 0
        _res_D_a[2, 4] = 0
        _res_D_a[2, 5] = 0
        _res_D_a[3, 0] = _a[2] * _tmp86 - _tmp49 * _tmp94 + _tmp56 * _tmp89 - _tmp64 * _tmp79
        _res_D_a[3, 1] = _a[3] * _tmp86 - _tmp42 * _tmp89 - _tmp49 * _tmp79 + _tmp64 * _tmp94
        _res_D_a[3, 2] = -_a[0] * _tmp86 - _tmp42 * _tmp79 + _tmp49 * _tmp89 + _tmp56 * _tmp94
        _res_D_a[3, 3] = _tmp4 + _tmp95
        _res_D_a[3, 4] = _tmp22 + _tmp96
        _res_D_a[3, 5] = _tmp15 + _tmp97
        _res_D_a[4, 0] = -_tmp100 * _tmp49 + _tmp104 * _tmp56 + _tmp106 * _tmp42 - _tmp109 * _tmp64
        _res_D_a[4, 1] = _tmp100 * _tmp64 - _tmp104 * _tmp42 + _tmp106 * _tmp56 - _tmp109 * _tmp49
        _res_D_a[4, 2] = _tmp100 * _tmp56 + _tmp104 * _tmp49 - _tmp106 * _tmp64 - _tmp109 * _tmp42
        _res_D_a[4, 3] = _tmp12 + _tmp96
        _res_D_a[4, 4] = _tmp19 + _tmp95
        _res_D_a[4, 5] = _tmp110 + _tmp28
        _res_D_a[5, 0] = _tmp111 * _tmp42 - _tmp112 * _tmp64 - _tmp113 * _tmp49 + _tmp114 * _tmp56
        _res_D_a[5, 1] = _tmp111 * _tmp56 - _tmp112 * _tmp49 + _tmp113 * _tmp64 - _tmp114 * _tmp42
        _res_D_a[5, 2] = -_tmp111 * _tmp64 - _tmp112 * _tmp42 + _tmp113 * _tmp56 + _tmp114 * _tmp49
        _res_D_a[5, 3] = _tmp16 + _tmp97
        _res_D_a[5, 4] = _tmp110 + _tmp24
        _res_D_a[5, 5] = _tmp19 + _tmp4 - 1
        _res_D_b = numpy.zeros((6, 6))
        _res_D_b[0, 0] = _tmp121 + _tmp126 * _tmp127 + _tmp137
        _res_D_b[0, 1] = -_tmp120 * _tmp135 + _tmp126 * _tmp129 + _tmp138 - _tmp139
        _res_D_b[0, 2] = _tmp119 * _tmp129 - _tmp126 * _tmp136 + _tmp140 - _tmp141
        _res_D_b[0, 3] = 0
        _res_D_b[0, 4] = 0
        _res_D_b[0, 5] = 0
        _res_D_b[1, 0] = -_tmp120 * _tmp142 + _tmp129 * _tmp143 - _tmp138 + _tmp139
        _res_D_b[1, 1] = _tmp121 + _tmp130 + _tmp136 * _tmp142 + _tmp145
        _res_D_b[1, 2] = -_tmp119 * _tmp127 + _tmp129 * _tmp142 - _tmp146 + _tmp147
        _res_D_b[1, 3] = 0
        _res_D_b[1, 4] = 0
        _res_D_b[1, 5] = 0
        _res_D_b[2, 0] = _tmp129 * _tmp148 - _tmp136 * _tmp143 - _tmp140 + _tmp141
        _res_D_b[2, 1] = -_tmp127 * _tmp148 + _tmp129 * _tmp135 + _tmp146 - _tmp147
        _res_D_b[2, 2] = _tmp120 * _tmp148 + _tmp137 + _tmp145
        _res_D_b[2, 3] = 0
        _res_D_b[2, 4] = 0
        _res_D_b[2, 5] = 0
        _res_D_b[3, 0] = 0
        _res_D_b[3, 1] = 0
        _res_D_b[3, 2] = 0
        _res_D_b[3, 3] = _tmp8
        _res_D_b[3, 4] = _tmp13
        _res_D_b[3, 5] = _tmp18
        _res_D_b[4, 0] = 0
        _res_D_b[4, 1] = 0
        _res_D_b[4, 2] = 0
        _res_D_b[4, 3] = _tmp23
        _res_D_b[4, 4] = _tmp21
        _res_D_b[4, 5] = _tmp26
        _res_D_b[5, 0] = 0
        _res_D_b[5, 1] = 0
        _res_D_b[5, 2] = 0
        _res_D_b[5, 3] = _tmp30
        _res_D_b[5, 4] = _tmp29
        _res_D_b[5, 5] = _tmp27
        return _res, _res_D_a, _res_D_b
