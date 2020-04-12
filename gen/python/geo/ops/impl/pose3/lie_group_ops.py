import numpy


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geometry.pose3.Pose3'>.
    """

    @staticmethod
    def expmap(vec, epsilon):
        # Input arrays

        # Output array
        res = [0.] * 7

        # Intermediate terms (19)
        _tmp0 = vec[2]**2
        _tmp1 = vec[1]**2
        _tmp2 = vec[0]**2
        _tmp3 = _tmp0 + _tmp1 + _tmp2 + epsilon**2
        _tmp4 = numpy.sqrt(_tmp3)
        _tmp5 = (1./2.)*_tmp4
        _tmp6 = numpy.sin(_tmp5)/_tmp4
        _tmp7 = (_tmp4 - numpy.sin(_tmp4))/_tmp3**(3./2.)
        _tmp8 = _tmp7*vec[0]*vec[2]
        _tmp9 = (1 - numpy.cos(_tmp4))/_tmp3
        _tmp10 = _tmp9*vec[1]
        _tmp11 = _tmp7*vec[1]
        _tmp12 = _tmp11*vec[0]
        _tmp13 = _tmp9*vec[2]
        _tmp14 = -_tmp1
        _tmp15 = -_tmp0
        _tmp16 = _tmp11*vec[2]
        _tmp17 = _tmp9*vec[0]
        _tmp18 = -_tmp2

        # Output terms (7)
        res[0] = _tmp6*vec[0]
        res[1] = _tmp6*vec[1]
        res[2] = _tmp6*vec[2]
        res[3] = numpy.cos(_tmp5)
        res[4] = vec[3]*(_tmp7*(_tmp14 + _tmp15) + 1) + vec[4]*(_tmp12 - _tmp13) + vec[5]*(_tmp10 + _tmp8)
        res[5] = vec[3]*(_tmp12 + _tmp13) + vec[4]*(_tmp7*(_tmp15 + _tmp18) + 1) + vec[5]*(_tmp16 - _tmp17)
        res[6] = vec[3]*(-_tmp10 + _tmp8) + vec[4]*(_tmp16 + _tmp17) + vec[5]*(_tmp7*(_tmp14 + _tmp18) + 1)

        return res

    @staticmethod
    def logmap(a, epsilon):
        # Input arrays
        _a = a.storage

        # Output array
        res = [0.] * 6

        # Intermediate terms (28)
        _tmp0 = _a[0]**2
        _tmp1 = _a[1]**2
        _tmp2 = _a[2]**2
        _tmp3 = _tmp0 + _tmp1 + _tmp2 + epsilon
        _tmp4 = numpy.sqrt(_tmp3)
        _tmp5 = _tmp4**(-1.0)
        _tmp6 = numpy.arctan(_tmp4/(_a[3] + epsilon))
        _tmp7 = _tmp5*_tmp6
        _tmp8 = 2*_tmp7
        _tmp9 = 4*_tmp6**2/_tmp3
        _tmp10 = _tmp1*_tmp9
        _tmp11 = -_tmp10
        _tmp12 = _tmp2*_tmp9
        _tmp13 = -_tmp12
        _tmp14 = _tmp0*_tmp9
        _tmp15 = _tmp10 + _tmp12 + _tmp14 + epsilon
        _tmp16 = numpy.sqrt(_tmp15)
        _tmp17 = 0.5*_tmp16
        _tmp18 = (-1./2.*_tmp16*numpy.cos(_tmp17)/numpy.sin(_tmp17) + 1)/_tmp15
        _tmp19 = _a[0]*_tmp18*_tmp9
        _tmp20 = _a[2]*_tmp19
        _tmp21 = 1.0*_tmp7
        _tmp22 = _a[1]*_tmp21
        _tmp23 = _a[1]*_tmp19
        _tmp24 = 1.0*_a[2]*_tmp5*_tmp6
        _tmp25 = _a[1]*_a[2]*_tmp18*_tmp9
        _tmp26 = _a[0]*_tmp21
        _tmp27 = -_tmp14

        # Output terms (6)
        res[0] = _a[0]*_tmp8
        res[1] = _a[1]*_tmp8
        res[2] = _a[2]*_tmp8
        res[3] = _a[4]*(_tmp18*(_tmp11 + _tmp13) + 1.0) + _a[5]*(_tmp23 + _tmp24) + _a[6]*(_tmp20 - _tmp22)
        res[4] = _a[4]*(_tmp23 - _tmp24) + _a[5]*(_tmp18*(_tmp13 + _tmp27) + 1.0) + _a[6]*(_tmp25 + _tmp26)
        res[5] = _a[4]*(_tmp20 + _tmp22) + _a[5]*(_tmp25 - _tmp26) + _a[6]*(_tmp18*(_tmp11 + _tmp27) + 1.0)

        return res

    @staticmethod
    def retract(a, vec, epsilon):
        # Input arrays
        _a = a.storage

        # Output array
        res = [0.] * 7

        # Intermediate terms (39)
        _tmp0 = vec[2]**2
        _tmp1 = vec[1]**2
        _tmp2 = vec[0]**2
        _tmp3 = _tmp0 + _tmp1 + _tmp2 + epsilon**2
        _tmp4 = numpy.sqrt(_tmp3)
        _tmp5 = (1./2.)*_tmp4
        _tmp6 = numpy.sin(_tmp5)/_tmp4
        _tmp7 = _a[1]*_tmp6
        _tmp8 = _a[2]*_tmp6
        _tmp9 = _tmp6*vec[0]
        _tmp10 = numpy.cos(_tmp5)
        _tmp11 = _tmp6*vec[2]
        _tmp12 = _tmp6*vec[1]
        _tmp13 = 2*_a[2]
        _tmp14 = _a[3]*_tmp13
        _tmp15 = 2*_a[0]*_a[1]
        _tmp16 = (_tmp4 - numpy.sin(_tmp4))/_tmp3**(3./2.)
        _tmp17 = _tmp16*vec[1]
        _tmp18 = _tmp17*vec[2]
        _tmp19 = (1 - numpy.cos(_tmp4))/_tmp3
        _tmp20 = _tmp19*vec[0]
        _tmp21 = -_tmp2
        _tmp22 = -_tmp0
        _tmp23 = _tmp17*vec[0]
        _tmp24 = _tmp19*vec[2]
        _tmp25 = vec[3]*(_tmp23 + _tmp24) + vec[4]*(_tmp16*(_tmp21 + _tmp22) + 1) + vec[5]*(_tmp18 - _tmp20)
        _tmp26 = -2*_a[2]**2
        _tmp27 = 1 - 2*_a[1]**2
        _tmp28 = _tmp16*vec[0]*vec[2]
        _tmp29 = _tmp19*vec[1]
        _tmp30 = -_tmp1
        _tmp31 = vec[3]*(_tmp16*(_tmp22 + _tmp30) + 1) + vec[4]*(_tmp23 - _tmp24) + vec[5]*(_tmp28 + _tmp29)
        _tmp32 = 2*_a[3]
        _tmp33 = _a[1]*_tmp32
        _tmp34 = _a[0]*_tmp13
        _tmp35 = vec[3]*(_tmp28 - _tmp29) + vec[4]*(_tmp18 + _tmp20) + vec[5]*(_tmp16*(_tmp21 + _tmp30) + 1)
        _tmp36 = -2*_a[0]**2
        _tmp37 = _a[0]*_tmp32
        _tmp38 = _a[1]*_tmp13

        # Output terms (7)
        res[0] = _a[0]*_tmp10 + _a[3]*_tmp9 + _tmp7*vec[2] - _tmp8*vec[1]
        res[1] = -_a[0]*_tmp11 + _a[1]*_tmp10 + _a[3]*_tmp12 + _tmp8*vec[0]
        res[2] = _a[0]*_tmp12 + _a[2]*_tmp10 + _a[3]*_tmp11 - _tmp7*vec[0]
        res[3] = -_a[0]*_tmp9 - _a[1]*_tmp12 - _a[2]*_tmp11 + _a[3]*_tmp10
        res[4] = _a[4] + _tmp25*(-_tmp14 + _tmp15) + _tmp31*(_tmp26 + _tmp27) + _tmp35*(_tmp33 + _tmp34)
        res[5] = _a[5] + _tmp25*(_tmp26 + _tmp36 + 1) + _tmp31*(_tmp14 + _tmp15) + _tmp35*(-_tmp37 + _tmp38)
        res[6] = _a[6] + _tmp25*(_tmp37 + _tmp38) + _tmp31*(-_tmp33 + _tmp34) + _tmp35*(_tmp27 + _tmp36)

        return res

    @staticmethod
    def local_coordinates(a, b, epsilon):
        # Input arrays
        _a = a.storage
        _b = b.storage

        # Output array
        res = [0.] * 6

        # Intermediate terms (53)
        _tmp0 = -_a[0]*_b[3] - _a[1]*_b[2] + _a[2]*_b[1] + _a[3]*_b[0]
        _tmp1 = _tmp0**2
        _tmp2 = _a[0]*_b[2] - _a[1]*_b[3] - _a[2]*_b[0] + _a[3]*_b[1]
        _tmp3 = _tmp2**2
        _tmp4 = -_a[0]*_b[1] + _a[1]*_b[0] - _a[2]*_b[3] + _a[3]*_b[2]
        _tmp5 = _tmp4**2
        _tmp6 = _tmp1 + _tmp3 + _tmp5 + epsilon
        _tmp7 = numpy.sqrt(_tmp6)
        _tmp8 = numpy.arctan(_tmp7/(_a[0]*_b[0] + _a[1]*_b[1] + _a[2]*_b[2] + _a[3]*_b[3] + epsilon))
        _tmp9 = _tmp8/_tmp7
        _tmp10 = 2*_tmp9
        _tmp11 = 4*_tmp8**2/_tmp6
        _tmp12 = _tmp1*_tmp11
        _tmp13 = _tmp11*_tmp5
        _tmp14 = _tmp11*_tmp3
        _tmp15 = _tmp12 + _tmp13 + _tmp14 + epsilon
        _tmp16 = numpy.sqrt(_tmp15)
        _tmp17 = 0.5*_tmp16
        _tmp18 = (-1./2.*_tmp16*numpy.cos(_tmp17)/numpy.sin(_tmp17) + 1)/_tmp15
        _tmp19 = _tmp0*_tmp11*_tmp18*_tmp4
        _tmp20 = 1.0*_tmp9
        _tmp21 = _tmp2*_tmp20
        _tmp22 = -2*_a[1]**2
        _tmp23 = 1 - 2*_a[0]**2
        _tmp24 = _tmp22 + _tmp23
        _tmp25 = 2*_a[3]
        _tmp26 = _a[0]*_tmp25
        _tmp27 = 2*_a[1]*_a[2]
        _tmp28 = -_tmp26 + _tmp27
        _tmp29 = _a[1]*_tmp25
        _tmp30 = 2*_a[0]
        _tmp31 = _a[2]*_tmp30
        _tmp32 = _tmp29 + _tmp31
        _tmp33 = -_a[4]*_tmp32 - _a[5]*_tmp28 - _a[6]*_tmp24 + _b[4]*_tmp32 + _b[5]*_tmp28 + _b[6]*_tmp24
        _tmp34 = -_tmp14
        _tmp35 = -_tmp13
        _tmp36 = -_tmp29 + _tmp31
        _tmp37 = _a[2]*_tmp25
        _tmp38 = _a[1]*_tmp30
        _tmp39 = _tmp37 + _tmp38
        _tmp40 = -2*_a[2]**2
        _tmp41 = _tmp22 + _tmp40 + 1
        _tmp42 = -_a[4]*_tmp41 - _a[5]*_tmp39 - _a[6]*_tmp36 + _b[4]*_tmp41 + _b[5]*_tmp39 + _b[6]*_tmp36
        _tmp43 = _tmp11*_tmp18*_tmp2
        _tmp44 = _tmp0*_tmp43
        _tmp45 = _tmp20*_tmp4
        _tmp46 = _tmp23 + _tmp40
        _tmp47 = -_tmp37 + _tmp38
        _tmp48 = _tmp26 + _tmp27
        _tmp49 = -_a[4]*_tmp47 - _a[5]*_tmp46 - _a[6]*_tmp48 + _b[4]*_tmp47 + _b[5]*_tmp46 + _b[6]*_tmp48
        _tmp50 = _tmp4*_tmp43
        _tmp51 = _tmp0*_tmp20
        _tmp52 = -_tmp12

        # Output terms (6)
        res[0] = _tmp0*_tmp10
        res[1] = _tmp10*_tmp2
        res[2] = _tmp10*_tmp4
        res[3] = _tmp33*(_tmp19 - _tmp21) + _tmp42*(_tmp18*(_tmp34 + _tmp35) + 1.0) + _tmp49*(_tmp44 + _tmp45)
        res[4] = _tmp33*(_tmp50 + _tmp51) + _tmp42*(_tmp44 - _tmp45) + _tmp49*(_tmp18*(_tmp35 + _tmp52) + 1.0)
        res[5] = _tmp33*(_tmp18*(_tmp34 + _tmp52) + 1.0) + _tmp42*(_tmp19 + _tmp21) + _tmp49*(_tmp50 - _tmp51)

        return res

