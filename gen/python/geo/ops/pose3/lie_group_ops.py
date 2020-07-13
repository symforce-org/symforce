import numpy


class LieGroupOps(object):
    """
    Python LieGroupOps implementatino for <class 'symforce.geo.pose3.Pose3'>.
    """

    @staticmethod
    def from_tangent(vec, epsilon):

        # Input arrays

        # Intermediate terms
        _tmp0 = vec[0]**2
        _tmp1 = vec[1]**2
        _tmp2 = vec[2]**2
        _tmp3 = _tmp0 + _tmp1 + _tmp2 + epsilon**2
        _tmp4 = numpy.sqrt(_tmp3)
        _tmp5 = (1./2.)*_tmp4
        _tmp6 = numpy.sin(_tmp5)/_tmp4
        _tmp7 = -_tmp2
        _tmp8 = -_tmp1
        _tmp9 = (_tmp4 - numpy.sin(_tmp4))/_tmp3**(3./2.)
        _tmp10 = (-numpy.cos(_tmp4) + 1)/_tmp3
        _tmp11 = _tmp10*vec[2]
        _tmp12 = _tmp9*vec[0]
        _tmp13 = _tmp12*vec[1]
        _tmp14 = _tmp10*vec[1]
        _tmp15 = _tmp12*vec[2]
        _tmp16 = -_tmp0
        _tmp17 = _tmp10*vec[0]
        _tmp18 = _tmp9*vec[1]*vec[2]

        # Output terms
        _res = [0.] * 7
        _res[0] = _tmp6*vec[0]
        _res[1] = _tmp6*vec[1]
        _res[2] = _tmp6*vec[2]
        _res[3] = numpy.cos(_tmp5)
        _res[4] = vec[3]*(_tmp9*(_tmp7 + _tmp8) + 1) + vec[4]*(-_tmp11 + _tmp13) + vec[5]*(_tmp14 + _tmp15)
        _res[5] = vec[3]*(_tmp11 + _tmp13) + vec[4]*(_tmp9*(_tmp16 + _tmp7) + 1) + vec[5]*(-_tmp17 + _tmp18)
        _res[6] = vec[3]*(-_tmp14 + _tmp15) + vec[4]*(_tmp17 + _tmp18) + vec[5]*(_tmp9*(_tmp16 + _tmp8) + 1)
        return _res

    @staticmethod
    def to_tangent(a, epsilon):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = numpy.arccos(numpy.amax((epsilon - 1,numpy.amin((_a[3],-epsilon + 1)))))
        _tmp1 = numpy.amax((epsilon,-_a[3]**2 + 1))
        _tmp2 = _tmp0/numpy.sqrt(_tmp1)
        _tmp3 = 2*_tmp2
        _tmp4 = 4*_tmp0**2/_tmp1
        _tmp5 = _a[2]**2*_tmp4
        _tmp6 = -_tmp5
        _tmp7 = _a[1]**2*_tmp4
        _tmp8 = -_tmp7
        _tmp9 = _a[0]**2*_tmp4
        _tmp10 = _tmp5 + _tmp7 + _tmp9 + epsilon
        _tmp11 = numpy.sqrt(_tmp10)
        _tmp12 = 0.5*_tmp11
        _tmp13 = (-1./2.*_tmp11*numpy.cos(_tmp12)/numpy.sin(_tmp12) + 1)/_tmp10
        _tmp14 = 1.0*_tmp2
        _tmp15 = _a[2]*_tmp14
        _tmp16 = _a[1]*_tmp13*_tmp4
        _tmp17 = _a[0]*_tmp16
        _tmp18 = _a[1]*_tmp14
        _tmp19 = _a[0]*_a[2]*_tmp13*_tmp4
        _tmp20 = -_tmp9
        _tmp21 = _a[0]*_tmp14
        _tmp22 = _a[2]*_tmp16

        # Output terms
        _res = [0.] * 6
        _res[0] = _a[0]*_tmp3
        _res[1] = _a[1]*_tmp3
        _res[2] = _a[2]*_tmp3
        _res[3] = _a[4]*(_tmp13*(_tmp6 + _tmp8) + 1.0) + _a[5]*(_tmp15 + _tmp17) + _a[6]*(-_tmp18 + _tmp19)
        _res[4] = _a[4]*(-_tmp15 + _tmp17) + _a[5]*(_tmp13*(_tmp20 + _tmp6) + 1.0) + _a[6]*(_tmp21 + _tmp22)
        _res[5] = _a[4]*(_tmp18 + _tmp19) + _a[5]*(-_tmp21 + _tmp22) + _a[6]*(_tmp13*(_tmp20 + _tmp8) + 1.0)
        return _res

    @staticmethod
    def retract(a, vec, epsilon):

        # Input arrays
        _a = a.data

        # Intermediate terms
        _tmp0 = vec[0]**2
        _tmp1 = vec[1]**2
        _tmp2 = vec[2]**2
        _tmp3 = _tmp0 + _tmp1 + _tmp2 + epsilon**2
        _tmp4 = numpy.sqrt(_tmp3)
        _tmp5 = (1./2.)*_tmp4
        _tmp6 = numpy.sin(_tmp5)/_tmp4
        _tmp7 = _tmp6*vec[0]
        _tmp8 = numpy.cos(_tmp5)
        _tmp9 = _tmp6*vec[1]
        _tmp10 = _tmp6*vec[2]
        _tmp11 = _a[3]*_tmp6
        _tmp12 = _a[0]*_tmp6
        _tmp13 = 2*_a[0]*_a[1]
        _tmp14 = 2*_a[2]
        _tmp15 = _a[3]*_tmp14
        _tmp16 = (-numpy.cos(_tmp4) + 1)/_tmp3
        _tmp17 = _tmp16*vec[2]
        _tmp18 = (_tmp4 - numpy.sin(_tmp4))/_tmp3**(3./2.)
        _tmp19 = _tmp18*vec[0]
        _tmp20 = _tmp19*vec[1]
        _tmp21 = -_tmp2
        _tmp22 = -_tmp0
        _tmp23 = _tmp16*vec[0]
        _tmp24 = _tmp18*vec[1]*vec[2]
        _tmp25 = vec[3]*(_tmp17 + _tmp20) + vec[4]*(_tmp18*(_tmp21 + _tmp22) + 1) + vec[5]*(-_tmp23 + _tmp24)
        _tmp26 = -2*_a[1]**2
        _tmp27 = -2*_a[2]**2 + 1
        _tmp28 = -_tmp1
        _tmp29 = _tmp16*vec[1]
        _tmp30 = _tmp19*vec[2]
        _tmp31 = vec[3]*(_tmp18*(_tmp21 + _tmp28) + 1) + vec[4]*(-_tmp17 + _tmp20) + vec[5]*(_tmp29 + _tmp30)
        _tmp32 = _a[0]*_tmp14
        _tmp33 = 2*_a[3]
        _tmp34 = _a[1]*_tmp33
        _tmp35 = vec[3]*(-_tmp29 + _tmp30) + vec[4]*(_tmp23 + _tmp24) + vec[5]*(_tmp18*(_tmp22 + _tmp28) + 1)
        _tmp36 = -2*_a[0]**2
        _tmp37 = _a[1]*_tmp14
        _tmp38 = _a[0]*_tmp33

        # Output terms
        _res = [0.] * 7
        _res[0] = _a[0]*_tmp8 + _a[1]*_tmp10 - _a[2]*_tmp9 + _a[3]*_tmp7
        _res[1] = _a[1]*_tmp8 + _a[2]*_tmp7 + _tmp11*vec[1] - _tmp12*vec[2]
        _res[2] = -_a[1]*_tmp7 + _a[2]*_tmp8 + _tmp11*vec[2] + _tmp12*vec[1]
        _res[3] = -_a[1]*_tmp9 - _a[2]*_tmp10 + _a[3]*_tmp8 - _tmp12*vec[0]
        _res[4] = _a[4] + _tmp25*(_tmp13 - _tmp15) + _tmp31*(_tmp26 + _tmp27) + _tmp35*(_tmp32 + _tmp34)
        _res[5] = _a[5] + _tmp25*(_tmp27 + _tmp36) + _tmp31*(_tmp13 + _tmp15) + _tmp35*(_tmp37 - _tmp38)
        _res[6] = _a[6] + _tmp25*(_tmp37 + _tmp38) + _tmp31*(_tmp32 - _tmp34) + _tmp35*(_tmp26 + _tmp36 + 1)
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
        _tmp8 = 4*_tmp2**2/_tmp3
        _tmp9 = _tmp7**2*_tmp8
        _tmp10 = -_tmp9
        _tmp11 = _tmp6**2*_tmp8
        _tmp12 = -_tmp11
        _tmp13 = _tmp0**2*_tmp8
        _tmp14 = _tmp11 + _tmp13 + _tmp9 + epsilon
        _tmp15 = numpy.sqrt(_tmp14)
        _tmp16 = 0.5*_tmp15
        _tmp17 = (-1./2.*_tmp15*numpy.cos(_tmp16)/numpy.sin(_tmp16) + 1)/_tmp14
        _tmp18 = -2*_a[2]**2
        _tmp19 = -2*_a[1]**2 + 1
        _tmp20 = _tmp18 + _tmp19
        _tmp21 = 2*_a[0]*_a[1]
        _tmp22 = 2*_a[2]
        _tmp23 = _a[3]*_tmp22
        _tmp24 = _tmp21 + _tmp23
        _tmp25 = _a[0]*_tmp22
        _tmp26 = 2*_a[3]
        _tmp27 = _a[1]*_tmp26
        _tmp28 = _tmp25 - _tmp27
        _tmp29 = -_a[4]*_tmp20 - _a[5]*_tmp24 - _a[6]*_tmp28 + _b[4]*_tmp20 + _b[5]*_tmp24 + _b[6]*_tmp28
        _tmp30 = 1.0*_tmp4
        _tmp31 = _tmp30*_tmp7
        _tmp32 = _tmp0*_tmp17*_tmp8
        _tmp33 = _tmp32*_tmp6
        _tmp34 = _tmp21 - _tmp23
        _tmp35 = -2*_a[0]**2
        _tmp36 = _tmp18 + _tmp35 + 1
        _tmp37 = _a[1]*_tmp22
        _tmp38 = _a[0]*_tmp26
        _tmp39 = _tmp37 + _tmp38
        _tmp40 = -_a[4]*_tmp34 - _a[5]*_tmp36 - _a[6]*_tmp39 + _b[4]*_tmp34 + _b[5]*_tmp36 + _b[6]*_tmp39
        _tmp41 = _tmp30*_tmp6
        _tmp42 = _tmp32*_tmp7
        _tmp43 = _tmp19 + _tmp35
        _tmp44 = _tmp25 + _tmp27
        _tmp45 = _tmp37 - _tmp38
        _tmp46 = -_a[4]*_tmp44 - _a[5]*_tmp45 - _a[6]*_tmp43 + _b[4]*_tmp44 + _b[5]*_tmp45 + _b[6]*_tmp43
        _tmp47 = -_tmp13
        _tmp48 = _tmp0*_tmp30
        _tmp49 = _tmp17*_tmp6*_tmp7*_tmp8

        # Output terms
        _res = [0.] * 6
        _res[0] = _tmp0*_tmp5
        _res[1] = _tmp5*_tmp6
        _res[2] = _tmp5*_tmp7
        _res[3] = _tmp29*(_tmp17*(_tmp10 + _tmp12) + 1.0) + _tmp40*(_tmp31 + _tmp33) + _tmp46*(-_tmp41 + _tmp42)
        _res[4] = _tmp29*(-_tmp31 + _tmp33) + _tmp40*(_tmp17*(_tmp10 + _tmp47) + 1.0) + _tmp46*(_tmp48 + _tmp49)
        _res[5] = _tmp29*(_tmp41 + _tmp42) + _tmp40*(-_tmp48 + _tmp49) + _tmp46*(_tmp17*(_tmp12 + _tmp47) + 1.0)
        return _res

