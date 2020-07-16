import numpy


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geo.pose3.Pose3'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):

        # Input arrays

        # Intermediate terms
        _tmp0 = vec[2]**2
        _tmp1 = vec[1]**2
        _tmp2 = vec[0]**2
        _tmp3 = _tmp0 + _tmp1 + _tmp2 + epsilon**2
        _tmp4 = numpy.sqrt(_tmp3)
        _tmp5 = (1./2.)*_tmp4
        _tmp6 = numpy.sin(_tmp5)/_tmp4
        _tmp7 = (_tmp4 - numpy.sin(_tmp4))/_tmp3**(3./2.)
        _tmp8 = _tmp7*vec[0]
        _tmp9 = _tmp8*vec[2]
        _tmp10 = (-numpy.cos(_tmp4) + 1)/_tmp3
        _tmp11 = _tmp10*vec[1]
        _tmp12 = _tmp8*vec[1]
        _tmp13 = _tmp10*vec[2]
        _tmp14 = -_tmp1
        _tmp15 = -_tmp0
        _tmp16 = _tmp7*vec[1]*vec[2]
        _tmp17 = _tmp10*vec[0]
        _tmp18 = -_tmp2

        # Output terms
        _res = [0.] * 7
        _res[0] = _tmp6*vec[0]
        _res[1] = _tmp6*vec[1]
        _res[2] = _tmp6*vec[2]
        _res[3] = numpy.cos(_tmp5)
        _res[4] = vec[3]*(_tmp7*(_tmp14 + _tmp15) + 1) + vec[4]*(_tmp12 - _tmp13) + vec[5]*(_tmp11 + _tmp9)
        _res[5] = vec[3]*(_tmp12 + _tmp13) + vec[4]*(_tmp7*(_tmp15 + _tmp18) + 1) + vec[5]*(_tmp16 - _tmp17)
        _res[6] = vec[3]*(-_tmp11 + _tmp9) + vec[4]*(_tmp16 + _tmp17) + vec[5]*(_tmp7*(_tmp14 + _tmp18) + 1)
        return _res

    @staticmethod
    def to_tangent(a, epsilon):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = numpy.amax((epsilon,-_a[3]**2 + 1))
        _tmp1 = numpy.arccos(numpy.amax((epsilon - 1,numpy.amin((_a[3],-epsilon + 1)))))
        _tmp2 = _tmp1/numpy.sqrt(_tmp0)
        _tmp3 = 2*_tmp2
        _tmp4 = 4*_tmp1**2/_tmp0
        _tmp5 = _a[2]**2*_tmp4
        _tmp6 = _a[1]**2*_tmp4
        _tmp7 = _a[0]**2*_tmp4
        _tmp8 = _tmp5 + _tmp6 + _tmp7 + epsilon
        _tmp9 = numpy.sqrt(_tmp8)
        _tmp10 = 0.5*_tmp9
        _tmp11 = (-1./2.*_tmp9*numpy.cos(_tmp10)/numpy.sin(_tmp10) + 1)/_tmp8
        _tmp12 = _a[2]*_tmp11*_tmp4
        _tmp13 = _a[0]*_tmp12
        _tmp14 = 1.0*_tmp2
        _tmp15 = _a[1]*_tmp14
        _tmp16 = _a[0]*_a[1]*_tmp11*_tmp4
        _tmp17 = _a[2]*_tmp14
        _tmp18 = -_tmp6
        _tmp19 = -_tmp5
        _tmp20 = _a[1]*_tmp12
        _tmp21 = _a[0]*_tmp14
        _tmp22 = -_tmp7

        # Output terms
        _res = [0.] * 6
        _res[0] = _a[0]*_tmp3
        _res[1] = _a[1]*_tmp3
        _res[2] = _a[2]*_tmp3
        _res[3] = _a[4]*(_tmp11*(_tmp18 + _tmp19) + 1.0) + _a[5]*(_tmp16 + _tmp17) + _a[6]*(_tmp13 - _tmp15)
        _res[4] = _a[4]*(_tmp16 - _tmp17) + _a[5]*(_tmp11*(_tmp19 + _tmp22) + 1.0) + _a[6]*(_tmp20 + _tmp21)
        _res[5] = _a[4]*(_tmp13 + _tmp15) + _a[5]*(_tmp20 - _tmp21) + _a[6]*(_tmp11*(_tmp18 + _tmp22) + 1.0)
        return _res

    @staticmethod
    def retract(a, vec, epsilon):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = vec[2]**2
        _tmp1 = vec[1]**2
        _tmp2 = vec[0]**2
        _tmp3 = _tmp0 + _tmp1 + _tmp2 + epsilon**2
        _tmp4 = numpy.sqrt(_tmp3)
        _tmp5 = (1./2.)*_tmp4
        _tmp6 = numpy.sin(_tmp5)/_tmp4
        _tmp7 = _tmp6*vec[2]
        _tmp8 = _tmp6*vec[1]
        _tmp9 = numpy.cos(_tmp5)
        _tmp10 = _a[3]*_tmp6
        _tmp11 = _tmp6*vec[0]
        _tmp12 = 2*_a[1]
        _tmp13 = _a[3]*_tmp12
        _tmp14 = 2*_a[0]
        _tmp15 = _a[2]*_tmp14
        _tmp16 = -_tmp2
        _tmp17 = -_tmp1
        _tmp18 = (_tmp4 - numpy.sin(_tmp4))/_tmp3**(3./2.)
        _tmp19 = _tmp18*vec[1]*vec[2]
        _tmp20 = (-numpy.cos(_tmp4) + 1)/_tmp3
        _tmp21 = _tmp20*vec[0]
        _tmp22 = _tmp18*vec[0]
        _tmp23 = _tmp22*vec[2]
        _tmp24 = _tmp20*vec[1]
        _tmp25 = vec[3]*(_tmp23 - _tmp24) + vec[4]*(_tmp19 + _tmp21) + vec[5]*(_tmp18*(_tmp16 + _tmp17) + 1)
        _tmp26 = 2*_a[2]*_a[3]
        _tmp27 = _a[0]*_tmp12
        _tmp28 = -_tmp0
        _tmp29 = _tmp22*vec[1]
        _tmp30 = _tmp20*vec[2]
        _tmp31 = vec[3]*(_tmp29 + _tmp30) + vec[4]*(_tmp18*(_tmp16 + _tmp28) + 1) + vec[5]*(_tmp19 - _tmp21)
        _tmp32 = -2*_a[2]**2
        _tmp33 = -2*_a[1]**2 + 1
        _tmp34 = vec[3]*(_tmp18*(_tmp17 + _tmp28) + 1) + vec[4]*(_tmp29 - _tmp30) + vec[5]*(_tmp23 + _tmp24)
        _tmp35 = _a[3]*_tmp14
        _tmp36 = _a[2]*_tmp12
        _tmp37 = -2*_a[0]**2

        # Output terms
        _res = [0.] * 7
        _res[0] = _a[0]*_tmp9 + _a[1]*_tmp7 - _a[2]*_tmp8 + _tmp10*vec[0]
        _res[1] = -_a[0]*_tmp7 + _a[1]*_tmp9 + _a[2]*_tmp11 + _a[3]*_tmp8
        _res[2] = _a[0]*_tmp8 - _a[1]*_tmp11 + _a[2]*_tmp9 + _tmp10*vec[2]
        _res[3] = -_a[0]*_tmp11 - _a[1]*_tmp8 - _a[2]*_tmp7 + _a[3]*_tmp9
        _res[4] = _a[4] + _tmp25*(_tmp13 + _tmp15) + _tmp31*(-_tmp26 + _tmp27) + _tmp34*(_tmp32 + _tmp33)
        _res[5] = _a[5] + _tmp25*(-_tmp35 + _tmp36) + _tmp31*(_tmp32 + _tmp37 + 1) + _tmp34*(_tmp26 + _tmp27)
        _res[6] = _a[6] + _tmp25*(_tmp33 + _tmp37) + _tmp31*(_tmp35 + _tmp36) + _tmp34*(-_tmp13 + _tmp15)
        return _res

    @staticmethod
    def local_coordinates(a, b, epsilon):

        # Input arrays
        _a = a.data
        _b = b.data

        # Intermediate terms
        _tmp0 = -_a[0]*_b[3] - _a[1]*_b[2] + _a[2]*_b[1] + _a[3]*_b[0]
        _tmp1 = _a[0]*_b[0] + _a[1]*_b[1] + _a[2]*_b[2] + _a[3]*_b[3]
        _tmp2 = numpy.arccos(numpy.amax((epsilon - 1,numpy.amin((_tmp1,-epsilon + 1)))))
        _tmp3 = numpy.amax((epsilon,-_tmp1**2 + 1))
        _tmp4 = _tmp2/numpy.sqrt(_tmp3)
        _tmp5 = 2*_tmp4
        _tmp6 = _a[0]*_b[2] - _a[1]*_b[3] - _a[2]*_b[0] + _a[3]*_b[1]
        _tmp7 = -_a[0]*_b[1] + _a[1]*_b[0] - _a[2]*_b[3] + _a[3]*_b[2]
        _tmp8 = -2*_a[0]**2
        _tmp9 = -2*_a[1]**2 + 1
        _tmp10 = _tmp8 + _tmp9
        _tmp11 = 2*_a[0]
        _tmp12 = _a[3]*_tmp11
        _tmp13 = 2*_a[2]
        _tmp14 = _a[1]*_tmp13
        _tmp15 = -_tmp12 + _tmp14
        _tmp16 = 2*_a[1]*_a[3]
        _tmp17 = _a[2]*_tmp11
        _tmp18 = _tmp16 + _tmp17
        _tmp19 = -_a[4]*_tmp18 - _a[5]*_tmp15 - _a[6]*_tmp10 + _b[4]*_tmp18 + _b[5]*_tmp15 + _b[6]*_tmp10
        _tmp20 = 4*_tmp2**2/_tmp3
        _tmp21 = _tmp20*_tmp7**2
        _tmp22 = _tmp20*_tmp6**2
        _tmp23 = _tmp0**2*_tmp20
        _tmp24 = _tmp21 + _tmp22 + _tmp23 + epsilon
        _tmp25 = numpy.sqrt(_tmp24)
        _tmp26 = 0.5*_tmp25
        _tmp27 = (-1./2.*_tmp25*numpy.cos(_tmp26)/numpy.sin(_tmp26) + 1)/_tmp24
        _tmp28 = _tmp0*_tmp20*_tmp27
        _tmp29 = _tmp28*_tmp7
        _tmp30 = 1.0*_tmp4
        _tmp31 = _tmp30*_tmp6
        _tmp32 = _tmp12 + _tmp14
        _tmp33 = -2*_a[2]**2
        _tmp34 = _tmp33 + _tmp8 + 1
        _tmp35 = _a[3]*_tmp13
        _tmp36 = _a[1]*_tmp11
        _tmp37 = -_tmp35 + _tmp36
        _tmp38 = -_a[4]*_tmp37 - _a[5]*_tmp34 - _a[6]*_tmp32 + _b[4]*_tmp37 + _b[5]*_tmp34 + _b[6]*_tmp32
        _tmp39 = _tmp28*_tmp6
        _tmp40 = _tmp30*_tmp7
        _tmp41 = -_tmp16 + _tmp17
        _tmp42 = _tmp35 + _tmp36
        _tmp43 = _tmp33 + _tmp9
        _tmp44 = -_a[4]*_tmp43 - _a[5]*_tmp42 - _a[6]*_tmp41 + _b[4]*_tmp43 + _b[5]*_tmp42 + _b[6]*_tmp41
        _tmp45 = -_tmp22
        _tmp46 = -_tmp21
        _tmp47 = _tmp20*_tmp27*_tmp6*_tmp7
        _tmp48 = _tmp0*_tmp30
        _tmp49 = -_tmp23

        # Output terms
        _res = [0.] * 6
        _res[0] = _tmp0*_tmp5
        _res[1] = _tmp5*_tmp6
        _res[2] = _tmp5*_tmp7
        _res[3] = _tmp19*(_tmp29 - _tmp31) + _tmp38*(_tmp39 + _tmp40) + _tmp44*(_tmp27*(_tmp45 + _tmp46) + 1.0)
        _res[4] = _tmp19*(_tmp47 + _tmp48) + _tmp38*(_tmp27*(_tmp46 + _tmp49) + 1.0) + _tmp44*(_tmp39 - _tmp40)
        _res[5] = _tmp19*(_tmp27*(_tmp45 + _tmp49) + 1.0) + _tmp38*(_tmp47 - _tmp48) + _tmp44*(_tmp29 + _tmp31)
        return _res

