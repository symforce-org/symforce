def az_el_from_point(nav_T_cam, nav_t_point, epsilon):
    """
    Transform a nav point into azimuth / elevation angles in the
    camera frame.

    Args:
        nav_T_cam (geo.Pose3): camera pose in the world
        nav_t_point (geo.Matrix): nav point
        epsilon (Scalar): small number to avoid singularities

    Returns:
        geo.Matrix: (azimuth, elevation)
    """

    # Input arrays
    _nav_T_cam = nav_T_cam.data

    # Output array
    res = [0.] * 2

    # Intermediate terms (23)
    _tmp0 = 2*_nav_T_cam[3]
    _tmp1 = _nav_T_cam[0]*_tmp0
    _tmp2 = 2*_nav_T_cam[1]
    _tmp3 = _nav_T_cam[2]*_tmp2
    _tmp4 = _tmp1 + _tmp3
    _tmp5 = -2*_nav_T_cam[0]**2
    _tmp6 = -2*_nav_T_cam[2]**2 + 1
    _tmp7 = _tmp5 + _tmp6
    _tmp8 = _nav_T_cam[2]*_tmp0
    _tmp9 = _nav_T_cam[0]*_tmp2
    _tmp10 = -_tmp8 + _tmp9
    _tmp11 = -_nav_T_cam[4]*_tmp10 - _nav_T_cam[5]*_tmp7 - _nav_T_cam[6]*_tmp4 + _tmp10*nav_t_point[0] + _tmp4*nav_t_point[2] + _tmp7*nav_t_point[1]
    _tmp12 = _nav_T_cam[1]*_tmp0
    _tmp13 = 2*_nav_T_cam[0]*_nav_T_cam[2]
    _tmp14 = -_tmp12 + _tmp13
    _tmp15 = _tmp8 + _tmp9
    _tmp16 = -2*_nav_T_cam[1]**2
    _tmp17 = _tmp16 + _tmp6
    _tmp18 = -_nav_T_cam[4]*_tmp17 - _nav_T_cam[5]*_tmp15 - _nav_T_cam[6]*_tmp14 + _tmp14*nav_t_point[2] + _tmp15*nav_t_point[1] + _tmp17*nav_t_point[0]
    _tmp19 = _tmp16 + _tmp5 + 1
    _tmp20 = -_tmp1 + _tmp3
    _tmp21 = _tmp12 + _tmp13
    _tmp22 = -_nav_T_cam[4]*_tmp21 - _nav_T_cam[5]*_tmp20 - _nav_T_cam[6]*_tmp19 + _tmp19*nav_t_point[2] + _tmp20*nav_t_point[1] + _tmp21*nav_t_point[0]

    # Output terms (2)
    res[0] = numpy.arctan2(_tmp11, _tmp18 + epsilon*(numpy.sign(_tmp18) + 0.5))
    res[1] = -numpy.arccos(_tmp22/(epsilon + numpy.sqrt(_tmp11**2 + _tmp18**2 + _tmp22**2))) + (1./2.)*numpy.pi


    return res
