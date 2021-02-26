import numpy
import typing as T

import geo  # pylint: disable=unused-import


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geo.pose3.Pose3'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):
        # type: (T.Sequence[float], float) -> T.List[float]

        # Total ops: 70

        # Input arrays

        # Intermediate terms (19)
        _tmp0 = vec[2] ** 2
        _tmp1 = vec[1] ** 2
        _tmp2 = vec[0] ** 2
        _tmp3 = _tmp0 + _tmp1 + _tmp2 + epsilon ** 2
        _tmp4 = numpy.sqrt(_tmp3)
        _tmp5 = (1.0 / 2.0) * _tmp4
        _tmp6 = numpy.sin(_tmp5) / _tmp4
        _tmp7 = (_tmp4 - numpy.sin(_tmp4)) / _tmp3 ** (3.0 / 2.0)
        _tmp8 = _tmp7 * vec[0]
        _tmp9 = _tmp8 * vec[2]
        _tmp10 = (1 - numpy.cos(_tmp4)) / _tmp3
        _tmp11 = _tmp10 * vec[1]
        _tmp12 = _tmp8 * vec[1]
        _tmp13 = _tmp10 * vec[2]
        _tmp14 = -_tmp1
        _tmp15 = -_tmp0
        _tmp16 = _tmp7 * vec[1] * vec[2]
        _tmp17 = _tmp10 * vec[0]
        _tmp18 = -_tmp2

        # Output terms
        _res = [0.0] * 7
        _res[0] = _tmp6 * vec[0]
        _res[1] = _tmp6 * vec[1]
        _res[2] = _tmp6 * vec[2]
        _res[3] = numpy.cos(_tmp5)
        _res[4] = (
            vec[3] * (_tmp7 * (_tmp14 + _tmp15) + 1)
            + vec[4] * (_tmp12 - _tmp13)
            + vec[5] * (_tmp11 + _tmp9)
        )
        _res[5] = (
            vec[3] * (_tmp12 + _tmp13)
            + vec[4] * (_tmp7 * (_tmp15 + _tmp18) + 1)
            + vec[5] * (_tmp16 - _tmp17)
        )
        _res[6] = (
            vec[3] * (-_tmp11 + _tmp9)
            + vec[4] * (_tmp16 + _tmp17)
            + vec[5] * (_tmp7 * (_tmp14 + _tmp18) + 1)
        )
        return _res

    @staticmethod
    def to_tangent(a, epsilon):
        # type: (geo.Pose3, float) -> T.List[float]

        # Total ops: 93

        # Input arrays
        _a = a.data

        # Intermediate terms (26)
        _tmp0 = 2 * numpy.amin((0, numpy.sign(_a[3])), axis=0) + 1
        _tmp1 = numpy.amin((abs(_a[3]), 1 - epsilon), axis=0)
        _tmp2 = numpy.arccos(_tmp1)
        _tmp3 = 1 - _tmp1 ** 2
        _tmp4 = _tmp0 * _tmp2 / numpy.sqrt(_tmp3)
        _tmp5 = 2 * _tmp4
        _tmp6 = _a[2] * _tmp4
        _tmp7 = 4 * _tmp0 ** 2 * _tmp2 ** 2 / _tmp3
        _tmp8 = _a[2] ** 2 * _tmp7
        _tmp9 = _a[1] ** 2 * _tmp7
        _tmp10 = _a[0] ** 2 * _tmp7
        _tmp11 = _tmp10 + _tmp8 + _tmp9 + epsilon
        _tmp12 = numpy.sqrt(_tmp11)
        _tmp13 = 0.5 * _tmp12
        _tmp14 = (-1.0 / 2.0 * _tmp12 * numpy.cos(_tmp13) / numpy.sin(_tmp13) + 1) / _tmp11
        _tmp15 = _a[0] * _tmp14 * _tmp7
        _tmp16 = _a[2] * _tmp15
        _tmp17 = 1.0 * _tmp4
        _tmp18 = _a[1] * _tmp17
        _tmp19 = _a[1] * _tmp15
        _tmp20 = 1.0 * _tmp6
        _tmp21 = -_tmp9
        _tmp22 = -_tmp8
        _tmp23 = _a[1] * _a[2] * _tmp14 * _tmp7
        _tmp24 = _a[0] * _tmp17
        _tmp25 = -_tmp10

        # Output terms
        _res = [0.0] * 6
        _res[0] = _a[0] * _tmp5
        _res[1] = _a[1] * _tmp5
        _res[2] = 2 * _tmp6
        _res[3] = (
            _a[4] * (_tmp14 * (_tmp21 + _tmp22) + 1.0)
            + _a[5] * (_tmp19 + _tmp20)
            + _a[6] * (_tmp16 - _tmp18)
        )
        _res[4] = (
            _a[4] * (_tmp19 - _tmp20)
            + _a[5] * (_tmp14 * (_tmp22 + _tmp25) + 1.0)
            + _a[6] * (_tmp23 + _tmp24)
        )
        _res[5] = (
            _a[4] * (_tmp16 + _tmp18)
            + _a[5] * (_tmp23 - _tmp24)
            + _a[6] * (_tmp14 * (_tmp21 + _tmp25) + 1.0)
        )
        return _res

    @staticmethod
    def retract(a, vec, epsilon):
        # type: (geo.Pose3, T.Sequence[float], float) -> T.List[float]

        # Total ops: 150

        # Input arrays
        _a = a.data

        # Intermediate terms (38)
        _tmp0 = vec[2] ** 2
        _tmp1 = vec[1] ** 2
        _tmp2 = vec[0] ** 2
        _tmp3 = _tmp0 + _tmp1 + _tmp2 + epsilon ** 2
        _tmp4 = numpy.sqrt(_tmp3)
        _tmp5 = (1.0 / 2.0) * _tmp4
        _tmp6 = numpy.sin(_tmp5) / _tmp4
        _tmp7 = _tmp6 * vec[2]
        _tmp8 = _tmp6 * vec[1]
        _tmp9 = numpy.cos(_tmp5)
        _tmp10 = _a[3] * _tmp6
        _tmp11 = _tmp6 * vec[0]
        _tmp12 = 2 * _a[1]
        _tmp13 = _a[3] * _tmp12
        _tmp14 = 2 * _a[0]
        _tmp15 = _a[2] * _tmp14
        _tmp16 = -_tmp2
        _tmp17 = -_tmp1
        _tmp18 = (_tmp4 - numpy.sin(_tmp4)) / _tmp3 ** (3.0 / 2.0)
        _tmp19 = _tmp18 * vec[1] * vec[2]
        _tmp20 = (1 - numpy.cos(_tmp4)) / _tmp3
        _tmp21 = _tmp20 * vec[0]
        _tmp22 = _tmp18 * vec[0]
        _tmp23 = _tmp22 * vec[2]
        _tmp24 = _tmp20 * vec[1]
        _tmp25 = (
            vec[3] * (_tmp23 - _tmp24)
            + vec[4] * (_tmp19 + _tmp21)
            + vec[5] * (_tmp18 * (_tmp16 + _tmp17) + 1)
        )
        _tmp26 = 2 * _a[2] * _a[3]
        _tmp27 = _a[0] * _tmp12
        _tmp28 = -_tmp0
        _tmp29 = _tmp22 * vec[1]
        _tmp30 = _tmp20 * vec[2]
        _tmp31 = (
            vec[3] * (_tmp29 + _tmp30)
            + vec[4] * (_tmp18 * (_tmp16 + _tmp28) + 1)
            + vec[5] * (_tmp19 - _tmp21)
        )
        _tmp32 = -2 * _a[2] ** 2
        _tmp33 = 1 - 2 * _a[1] ** 2
        _tmp34 = (
            vec[3] * (_tmp18 * (_tmp17 + _tmp28) + 1)
            + vec[4] * (_tmp29 - _tmp30)
            + vec[5] * (_tmp23 + _tmp24)
        )
        _tmp35 = _a[3] * _tmp14
        _tmp36 = _a[2] * _tmp12
        _tmp37 = -2 * _a[0] ** 2

        # Output terms
        _res = [0.0] * 7
        _res[0] = _a[0] * _tmp9 + _a[1] * _tmp7 - _a[2] * _tmp8 + _tmp10 * vec[0]
        _res[1] = -_a[0] * _tmp7 + _a[1] * _tmp9 + _a[2] * _tmp11 + _a[3] * _tmp8
        _res[2] = _a[0] * _tmp8 - _a[1] * _tmp11 + _a[2] * _tmp9 + _tmp10 * vec[2]
        _res[3] = -_a[0] * _tmp11 - _a[1] * _tmp8 - _a[2] * _tmp7 + _a[3] * _tmp9
        _res[4] = (
            _a[4]
            + _tmp25 * (_tmp13 + _tmp15)
            + _tmp31 * (-_tmp26 + _tmp27)
            + _tmp34 * (_tmp32 + _tmp33)
        )
        _res[5] = (
            _a[5]
            + _tmp25 * (-_tmp35 + _tmp36)
            + _tmp31 * (_tmp32 + _tmp37 + 1)
            + _tmp34 * (_tmp26 + _tmp27)
        )
        _res[6] = (
            _a[6]
            + _tmp25 * (_tmp33 + _tmp37)
            + _tmp31 * (_tmp35 + _tmp36)
            + _tmp34 * (-_tmp13 + _tmp15)
        )
        return _res

    @staticmethod
    def local_coordinates(a, b, epsilon):
        # type: (geo.Pose3, geo.Pose3, float) -> T.List[float]

        # Total ops: 196

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms (53)
        _tmp0 = -_a[0] * _b[3] - _a[1] * _b[2] + _a[2] * _b[1] + _a[3] * _b[0]
        _tmp1 = _a[0] * _b[0] + _a[1] * _b[1] + _a[2] * _b[2] + _a[3] * _b[3]
        _tmp2 = numpy.amin((abs(_tmp1), 1 - epsilon), axis=0)
        _tmp3 = 1 - _tmp2 ** 2
        _tmp4 = 2 * numpy.amin((0, numpy.sign(_tmp1)), axis=0) + 1
        _tmp5 = numpy.arccos(_tmp2)
        _tmp6 = _tmp4 * _tmp5 / numpy.sqrt(_tmp3)
        _tmp7 = 2 * _tmp6
        _tmp8 = _a[0] * _b[2] - _a[1] * _b[3] - _a[2] * _b[0] + _a[3] * _b[1]
        _tmp9 = _tmp6 * _tmp8
        _tmp10 = -_a[0] * _b[1] + _a[1] * _b[0] - _a[2] * _b[3] + _a[3] * _b[2]
        _tmp11 = -2 * _a[0] ** 2
        _tmp12 = 1 - 2 * _a[1] ** 2
        _tmp13 = _tmp11 + _tmp12
        _tmp14 = 2 * _a[0]
        _tmp15 = _a[3] * _tmp14
        _tmp16 = 2 * _a[1]
        _tmp17 = _a[2] * _tmp16
        _tmp18 = -_tmp15 + _tmp17
        _tmp19 = _a[3] * _tmp16
        _tmp20 = _a[2] * _tmp14
        _tmp21 = _tmp19 + _tmp20
        _tmp22 = (
            -_a[4] * _tmp21
            - _a[5] * _tmp18
            - _a[6] * _tmp13
            + _b[4] * _tmp21
            + _b[5] * _tmp18
            + _b[6] * _tmp13
        )
        _tmp23 = 4 * _tmp4 ** 2 * _tmp5 ** 2 / _tmp3
        _tmp24 = _tmp23 * _tmp8 ** 2
        _tmp25 = _tmp10 ** 2 * _tmp23
        _tmp26 = _tmp0 ** 2 * _tmp23
        _tmp27 = _tmp24 + _tmp25 + _tmp26 + epsilon
        _tmp28 = numpy.sqrt(_tmp27)
        _tmp29 = 0.5 * _tmp28
        _tmp30 = (-1.0 / 2.0 * _tmp28 * numpy.cos(_tmp29) / numpy.sin(_tmp29) + 1) / _tmp27
        _tmp31 = _tmp0 * _tmp23 * _tmp30
        _tmp32 = _tmp10 * _tmp31
        _tmp33 = 1.0 * _tmp9
        _tmp34 = _tmp15 + _tmp17
        _tmp35 = -2 * _a[2] ** 2
        _tmp36 = _tmp11 + _tmp35 + 1
        _tmp37 = 2 * _a[2] * _a[3]
        _tmp38 = _a[1] * _tmp14
        _tmp39 = -_tmp37 + _tmp38
        _tmp40 = (
            -_a[4] * _tmp39
            - _a[5] * _tmp36
            - _a[6] * _tmp34
            + _b[4] * _tmp39
            + _b[5] * _tmp36
            + _b[6] * _tmp34
        )
        _tmp41 = _tmp31 * _tmp8
        _tmp42 = 1.0 * _tmp6
        _tmp43 = _tmp10 * _tmp42
        _tmp44 = -_tmp19 + _tmp20
        _tmp45 = _tmp37 + _tmp38
        _tmp46 = _tmp12 + _tmp35
        _tmp47 = (
            -_a[4] * _tmp46
            - _a[5] * _tmp45
            - _a[6] * _tmp44
            + _b[4] * _tmp46
            + _b[5] * _tmp45
            + _b[6] * _tmp44
        )
        _tmp48 = -_tmp24
        _tmp49 = -_tmp25
        _tmp50 = _tmp10 * _tmp23 * _tmp30 * _tmp8
        _tmp51 = _tmp0 * _tmp42
        _tmp52 = -_tmp26

        # Output terms
        _res = [0.0] * 6
        _res[0] = _tmp0 * _tmp7
        _res[1] = 2 * _tmp9
        _res[2] = _tmp10 * _tmp7
        _res[3] = (
            _tmp22 * (_tmp32 - _tmp33)
            + _tmp40 * (_tmp41 + _tmp43)
            + _tmp47 * (_tmp30 * (_tmp48 + _tmp49) + 1.0)
        )
        _res[4] = (
            _tmp22 * (_tmp50 + _tmp51)
            + _tmp40 * (_tmp30 * (_tmp49 + _tmp52) + 1.0)
            + _tmp47 * (_tmp41 - _tmp43)
        )
        _res[5] = (
            _tmp22 * (_tmp30 * (_tmp48 + _tmp52) + 1.0)
            + _tmp40 * (_tmp50 - _tmp51)
            + _tmp47 * (_tmp32 + _tmp33)
        )
        return _res
