import math
import numpy
import typing as T

import sym  # pylint: disable=unused-import


class GroupOps(object):
    """
    Python GroupOps implementation for <class 'symforce.geo.rot3.Rot3'>.
    """

    @staticmethod
    def identity():
        # type: () -> T.List[float]

        # Total ops: 0

        # Input arrays

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 4
        _res[0] = 0
        _res[1] = 0
        _res[2] = 0
        _res[3] = 1
        return _res

    @staticmethod
    def inverse(a):
        # type: (sym.Rot3) -> T.List[float]

        # Total ops: 3

        # Input arrays
        _a = a.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 4
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        return _res

    @staticmethod
    def compose(a, b):
        # type: (sym.Rot3, sym.Rot3) -> T.List[float]

        # Total ops: 32

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 4
        _res[0] = _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0]
        _res[1] = -_a[0] * _b[2] + _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1]
        _res[2] = _a[0] * _b[1] - _a[1] * _b[0] + _a[2] * _b[3] + _a[3] * _b[2]
        _res[3] = -_a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] + _a[3] * _b[3]
        return _res

    @staticmethod
    def between(a, b):
        # type: (sym.Rot3, sym.Rot3) -> T.List[float]

        # Total ops: 38

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (0)

        # Output terms
        _res = [0.0] * 4
        _res[0] = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0]
        _res[1] = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1]
        _res[2] = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2]
        _res[3] = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3]
        return _res

    @staticmethod
    def inverse_with_jacobian(a):
        # type: (sym.Rot3) -> T.Tuple[T.List[float], numpy.ndarray]

        # Total ops: 39

        # Input arrays
        _a = a.data

        # Intermediate terms (13)
        _tmp0 = _a[2] ** 2
        _tmp1 = _a[0] ** 2
        _tmp2 = -_a[3] ** 2
        _tmp3 = _a[1] ** 2
        _tmp4 = _tmp2 + _tmp3
        _tmp5 = 2 * _a[2]
        _tmp6 = _a[3] * _tmp5
        _tmp7 = -2 * _a[0] * _a[1]
        _tmp8 = 2 * _a[3]
        _tmp9 = _a[1] * _tmp8
        _tmp10 = -_a[0] * _tmp5
        _tmp11 = _a[0] * _tmp8
        _tmp12 = -_a[1] * _tmp5

        # Output terms
        _res = [0.0] * 4
        _res[0] = -_a[0]
        _res[1] = -_a[1]
        _res[2] = -_a[2]
        _res[3] = _a[3]
        _res_D_a = numpy.zeros((3, 3))
        _res_D_a[0, 0] = _tmp0 - _tmp1 + _tmp4
        _res_D_a[0, 1] = _tmp6 + _tmp7
        _res_D_a[0, 2] = _tmp10 - _tmp9
        _res_D_a[1, 0] = -_tmp6 + _tmp7
        _res_D_a[1, 1] = _tmp0 + _tmp1 + _tmp2 - _tmp3
        _res_D_a[1, 2] = _tmp11 + _tmp12
        _res_D_a[2, 0] = _tmp10 + _tmp9
        _res_D_a[2, 1] = -_tmp11 + _tmp12
        _res_D_a[2, 2] = -_tmp0 + _tmp1 + _tmp4
        return _res, _res_D_a

    @staticmethod
    def compose_with_jacobians(a, b):
        # type: (sym.Rot3, sym.Rot3) -> T.Tuple[T.List[float], numpy.ndarray, numpy.ndarray]

        # Total ops: 300

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (79)
        _tmp0 = _a[0] * _b[3] + _a[1] * _b[2] - _a[2] * _b[1] + _a[3] * _b[0]
        _tmp1 = -_a[0] * _b[2] + _a[1] * _b[3] + _a[2] * _b[0] + _a[3] * _b[1]
        _tmp2 = _a[0] * _b[1] - _a[1] * _b[0] + _a[2] * _b[3] + _a[3] * _b[2]
        _tmp3 = -_a[0] * _b[0] - _a[1] * _b[1] - _a[2] * _b[2] + _a[3] * _b[3]
        _tmp4 = 2 * _tmp2
        _tmp5 = _b[3] * _tmp4
        _tmp6 = 2 * _tmp0
        _tmp7 = _b[1] * _tmp6
        _tmp8 = 2 * _tmp1
        _tmp9 = _b[0] * _tmp8
        _tmp10 = 2 * _tmp3
        _tmp11 = _b[2] * _tmp10
        _tmp12 = _tmp11 + _tmp5 + _tmp7 + _tmp9
        _tmp13 = (1.0 / 2.0) * _a[2]
        _tmp14 = _b[2] * _tmp8
        _tmp15 = _b[0] * _tmp10
        _tmp16 = _b[1] * _tmp4
        _tmp17 = _b[3] * _tmp6
        _tmp18 = _tmp16 - _tmp17
        _tmp19 = -_tmp14 + _tmp15 + _tmp18
        _tmp20 = (1.0 / 2.0) * _a[0]
        _tmp21 = _b[2] * _tmp4
        _tmp22 = -_tmp21
        _tmp23 = _b[0] * _tmp6
        _tmp24 = _b[1] * _tmp8
        _tmp25 = -_tmp24
        _tmp26 = _b[3] * _tmp10
        _tmp27 = _tmp22 + _tmp23 + _tmp25 + _tmp26
        _tmp28 = (1.0 / 2.0) * _a[3]
        _tmp29 = _b[0] * _tmp4
        _tmp30 = _b[1] * _tmp10
        _tmp31 = _b[2] * _tmp6
        _tmp32 = _b[3] * _tmp8
        _tmp33 = _tmp31 - _tmp32
        _tmp34 = _tmp29 - _tmp30 + _tmp33
        _tmp35 = (1.0 / 2.0) * _a[1]
        _tmp36 = -_tmp5 + _tmp9
        _tmp37 = -_tmp11 + _tmp36 + _tmp7
        _tmp38 = _tmp14 + _tmp15 + _tmp16 + _tmp17
        _tmp39 = -_tmp23 + _tmp26
        _tmp40 = _tmp22 + _tmp24 + _tmp39
        _tmp41 = -_tmp29 + _tmp30 + _tmp33
        _tmp42 = _tmp11 + _tmp36 - _tmp7
        _tmp43 = _tmp14 - _tmp15 + _tmp18
        _tmp44 = _tmp21 + _tmp25 + _tmp39
        _tmp45 = _tmp29 + _tmp30 + _tmp31 + _tmp32
        _tmp46 = _a[0] * _tmp4
        _tmp47 = _a[2] * _tmp6
        _tmp48 = _a[3] * _tmp8
        _tmp49 = _a[1] * _tmp10
        _tmp50 = -_tmp46 + _tmp47 - _tmp48 + _tmp49
        _tmp51 = (1.0 / 2.0) * _b[1]
        _tmp52 = -_tmp50 * _tmp51
        _tmp53 = _a[3] * _tmp4
        _tmp54 = _a[1] * _tmp6
        _tmp55 = _a[0] * _tmp8
        _tmp56 = _a[2] * _tmp10
        _tmp57 = _tmp53 + _tmp54 - _tmp55 - _tmp56
        _tmp58 = (1.0 / 2.0) * _b[2]
        _tmp59 = _a[0] * _tmp6 + _a[1] * _tmp8 + _a[2] * _tmp4 + _a[3] * _tmp10
        _tmp60 = (1.0 / 2.0) * _b[3]
        _tmp61 = _tmp59 * _tmp60
        _tmp62 = _a[1] * _tmp4
        _tmp63 = _a[3] * _tmp6
        _tmp64 = _a[2] * _tmp8
        _tmp65 = _a[0] * _tmp10
        _tmp66 = _tmp62 - _tmp63 - _tmp64 + _tmp65
        _tmp67 = (1.0 / 2.0) * _b[0]
        _tmp68 = _tmp61 - _tmp66 * _tmp67
        _tmp69 = _tmp50 * _tmp67
        _tmp70 = _tmp58 * _tmp59
        _tmp71 = _tmp51 * _tmp59
        _tmp72 = _tmp58 * _tmp66
        _tmp73 = -_tmp53 - _tmp54 + _tmp55 + _tmp56
        _tmp74 = -_tmp62 + _tmp63 + _tmp64 - _tmp65
        _tmp75 = -_tmp58 * _tmp73
        _tmp76 = _tmp59 * _tmp67
        _tmp77 = _tmp51 * _tmp73
        _tmp78 = _tmp46 - _tmp47 + _tmp48 - _tmp49

        # Output terms
        _res = [0.0] * 4
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res[2] = _tmp2
        _res[3] = _tmp3
        _res_D_a = numpy.zeros((3, 3))
        _res_D_a[0, 0] = _tmp12 * _tmp13 - _tmp19 * _tmp20 + _tmp27 * _tmp28 - _tmp34 * _tmp35
        _res_D_a[0, 1] = _tmp12 * _tmp28 - _tmp13 * _tmp27 - _tmp19 * _tmp35 + _tmp20 * _tmp34
        _res_D_a[0, 2] = -_tmp12 * _tmp20 - _tmp13 * _tmp19 + _tmp27 * _tmp35 + _tmp28 * _tmp34
        _res_D_a[1, 0] = _tmp13 * _tmp40 - _tmp20 * _tmp41 + _tmp28 * _tmp37 - _tmp35 * _tmp38
        _res_D_a[1, 1] = -_tmp13 * _tmp37 + _tmp20 * _tmp38 + _tmp28 * _tmp40 - _tmp35 * _tmp41
        _res_D_a[1, 2] = -_tmp13 * _tmp41 - _tmp20 * _tmp40 + _tmp28 * _tmp38 + _tmp35 * _tmp37
        _res_D_a[2, 0] = _tmp13 * _tmp43 - _tmp20 * _tmp42 + _tmp28 * _tmp45 - _tmp35 * _tmp44
        _res_D_a[2, 1] = -_tmp13 * _tmp45 + _tmp20 * _tmp44 + _tmp28 * _tmp43 - _tmp35 * _tmp42
        _res_D_a[2, 2] = -_tmp13 * _tmp42 - _tmp20 * _tmp43 + _tmp28 * _tmp44 + _tmp35 * _tmp45
        _res_D_b = numpy.zeros((3, 3))
        _res_D_b[0, 0] = _tmp52 + _tmp57 * _tmp58 + _tmp68
        _res_D_b[0, 1] = -_tmp51 * _tmp66 + _tmp57 * _tmp60 + _tmp69 - _tmp70
        _res_D_b[0, 2] = _tmp50 * _tmp60 - _tmp57 * _tmp67 + _tmp71 - _tmp72
        _res_D_b[1, 0] = -_tmp51 * _tmp74 + _tmp60 * _tmp73 - _tmp69 + _tmp70
        _res_D_b[1, 1] = _tmp52 + _tmp61 + _tmp67 * _tmp74 + _tmp75
        _res_D_b[1, 2] = -_tmp50 * _tmp58 + _tmp60 * _tmp74 - _tmp76 + _tmp77
        _res_D_b[2, 0] = _tmp60 * _tmp78 - _tmp67 * _tmp73 - _tmp71 + _tmp72
        _res_D_b[2, 1] = -_tmp58 * _tmp78 + _tmp60 * _tmp66 + _tmp76 - _tmp77
        _res_D_b[2, 2] = _tmp51 * _tmp78 + _tmp68 + _tmp75
        return _res, _res_D_a, _res_D_b

    @staticmethod
    def between_with_jacobians(a, b):
        # type: (sym.Rot3, sym.Rot3) -> T.Tuple[T.List[float], numpy.ndarray, numpy.ndarray]

        # Total ops: 315

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (82)
        _tmp0 = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0]
        _tmp1 = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1]
        _tmp2 = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2]
        _tmp3 = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3]
        _tmp4 = 2 * _b[2]
        _tmp5 = _tmp2 * _tmp4
        _tmp6 = 2 * _tmp0
        _tmp7 = _b[0] * _tmp6
        _tmp8 = 2 * _tmp1
        _tmp9 = _b[1] * _tmp8
        _tmp10 = 2 * _tmp3
        _tmp11 = -_b[3] * _tmp10
        _tmp12 = _tmp11 + _tmp5 - _tmp7 + _tmp9
        _tmp13 = (1.0 / 2.0) * _a[3]
        _tmp14 = 2 * _tmp2
        _tmp15 = _b[3] * _tmp14
        _tmp16 = -_tmp15
        _tmp17 = -_b[1] * _tmp6
        _tmp18 = _b[0] * _tmp8
        _tmp19 = -_tmp18
        _tmp20 = _tmp3 * _tmp4
        _tmp21 = _tmp16 + _tmp17 + _tmp19 - _tmp20
        _tmp22 = (1.0 / 2.0) * _a[2]
        _tmp23 = _b[1] * _tmp14
        _tmp24 = _b[0] * _tmp10
        _tmp25 = _b[3] * _tmp6
        _tmp26 = -_tmp1 * _tmp4
        _tmp27 = -_tmp25 + _tmp26
        _tmp28 = _tmp23 + _tmp24 + _tmp27
        _tmp29 = (1.0 / 2.0) * _a[0]
        _tmp30 = _b[3] * _tmp8
        _tmp31 = _b[1] * _tmp10
        _tmp32 = -_b[0] * _tmp14
        _tmp33 = _tmp0 * _tmp4
        _tmp34 = _tmp32 - _tmp33
        _tmp35 = _tmp30 + _tmp31 + _tmp34
        _tmp36 = (1.0 / 2.0) * _a[1]
        _tmp37 = _tmp11 + _tmp7
        _tmp38 = _tmp37 + _tmp5 - _tmp9
        _tmp39 = _tmp17 + _tmp20
        _tmp40 = _tmp15 + _tmp19 + _tmp39
        _tmp41 = -_tmp23
        _tmp42 = -_tmp24 + _tmp27 + _tmp41
        _tmp43 = -_tmp30
        _tmp44 = _tmp31 + _tmp32 + _tmp33 + _tmp43
        _tmp45 = _tmp37 - _tmp5 + _tmp9
        _tmp46 = _tmp16 + _tmp18 + _tmp39
        _tmp47 = _tmp24 + _tmp25 + _tmp26 + _tmp41
        _tmp48 = -_tmp31 + _tmp34 + _tmp43
        _tmp49 = _a[3] * _tmp14
        _tmp50 = _a[1] * _tmp6
        _tmp51 = _a[0] * _tmp8
        _tmp52 = _a[2] * _tmp10
        _tmp53 = _tmp49 - _tmp50 + _tmp51 + _tmp52
        _tmp54 = (1.0 / 2.0) * _b[2]
        _tmp55 = _a[0] * _tmp14
        _tmp56 = _a[2] * _tmp6
        _tmp57 = _a[3] * _tmp8
        _tmp58 = _a[1] * _tmp10
        _tmp59 = (1.0 / 2.0) * _tmp55 - 1.0 / 2.0 * _tmp56 - 1.0 / 2.0 * _tmp57 - 1.0 / 2.0 * _tmp58
        _tmp60 = -_b[1] * _tmp59
        _tmp61 = -_a[0] * _tmp6 - _a[1] * _tmp8 - _a[2] * _tmp14 + _a[3] * _tmp10
        _tmp62 = (1.0 / 2.0) * _b[3]
        _tmp63 = _tmp61 * _tmp62
        _tmp64 = _a[1] * _tmp14
        _tmp65 = _a[3] * _tmp6
        _tmp66 = _a[2] * _tmp8
        _tmp67 = _a[0] * _tmp10
        _tmp68 = -_tmp64 - _tmp65 + _tmp66 - _tmp67
        _tmp69 = (1.0 / 2.0) * _b[0]
        _tmp70 = _tmp63 - _tmp68 * _tmp69
        _tmp71 = _tmp54 * _tmp61
        _tmp72 = (1.0 / 2.0) * _b[1]
        _tmp73 = _b[0] * _tmp59
        _tmp74 = _tmp61 * _tmp72
        _tmp75 = _tmp54 * _tmp68
        _tmp76 = -_tmp49 + _tmp50 - _tmp51 - _tmp52
        _tmp77 = _tmp64 + _tmp65 - _tmp66 + _tmp67
        _tmp78 = -_tmp54 * _tmp76
        _tmp79 = _tmp61 * _tmp69
        _tmp80 = _tmp72 * _tmp76
        _tmp81 = -_tmp55 + _tmp56 + _tmp57 + _tmp58

        # Output terms
        _res = [0.0] * 4
        _res[0] = _tmp0
        _res[1] = _tmp1
        _res[2] = _tmp2
        _res[3] = _tmp3
        _res_D_a = numpy.zeros((3, 3))
        _res_D_a[0, 0] = _tmp12 * _tmp13 + _tmp21 * _tmp22 - _tmp28 * _tmp29 - _tmp35 * _tmp36
        _res_D_a[0, 1] = -_tmp12 * _tmp22 + _tmp13 * _tmp21 - _tmp28 * _tmp36 + _tmp29 * _tmp35
        _res_D_a[0, 2] = _tmp12 * _tmp36 + _tmp13 * _tmp35 - _tmp21 * _tmp29 - _tmp22 * _tmp28
        _res_D_a[1, 0] = _tmp13 * _tmp40 + _tmp22 * _tmp38 - _tmp29 * _tmp44 - _tmp36 * _tmp42
        _res_D_a[1, 1] = _tmp13 * _tmp38 - _tmp22 * _tmp40 + _tmp29 * _tmp42 - _tmp36 * _tmp44
        _res_D_a[1, 2] = _tmp13 * _tmp42 - _tmp22 * _tmp44 - _tmp29 * _tmp38 + _tmp36 * _tmp40
        _res_D_a[2, 0] = _tmp13 * _tmp48 + _tmp22 * _tmp47 - _tmp29 * _tmp46 - _tmp36 * _tmp45
        _res_D_a[2, 1] = _tmp13 * _tmp47 - _tmp22 * _tmp48 + _tmp29 * _tmp45 - _tmp36 * _tmp46
        _res_D_a[2, 2] = _tmp13 * _tmp45 - _tmp22 * _tmp46 - _tmp29 * _tmp47 + _tmp36 * _tmp48
        _res_D_b = numpy.zeros((3, 3))
        _res_D_b[0, 0] = _tmp53 * _tmp54 + _tmp60 + _tmp70
        _res_D_b[0, 1] = _tmp53 * _tmp62 - _tmp68 * _tmp72 - _tmp71 + _tmp73
        _res_D_b[0, 2] = _b[3] * _tmp59 - _tmp53 * _tmp69 + _tmp74 - _tmp75
        _res_D_b[1, 0] = _tmp62 * _tmp76 + _tmp71 - _tmp72 * _tmp77 - _tmp73
        _res_D_b[1, 1] = _tmp60 + _tmp63 + _tmp69 * _tmp77 + _tmp78
        _res_D_b[1, 2] = -_b[2] * _tmp59 + _tmp62 * _tmp77 - _tmp79 + _tmp80
        _res_D_b[2, 0] = _tmp62 * _tmp81 - _tmp69 * _tmp76 - _tmp74 + _tmp75
        _res_D_b[2, 1] = -_tmp54 * _tmp81 + _tmp62 * _tmp68 + _tmp79 - _tmp80
        _res_D_b[2, 2] = _tmp70 + _tmp72 * _tmp81 + _tmp78
        return _res, _res_D_a, _res_D_b
