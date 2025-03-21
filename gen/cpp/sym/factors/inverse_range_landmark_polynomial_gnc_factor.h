// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

#include <sym/polynomial_camera_cal.h>
#include <sym/pose3.h>

namespace sym {

/**
 * Return the 2dof residual of reprojecting the landmark ray into the target spherical camera and
 * comparing it against the correspondence.
 *
 * The landmark is specified as a camera point in the source camera with an inverse range; this
 * means the landmark is fixed in the source camera and always has residual 0 there (this 0
 * residual is not returned, only the residual in the target camera is returned).
 *
 * The norm of the residual is whitened using the
 * :class:`BarronNoiseModel <symforce.opt.noise_models.BarronNoiseModel>`.  Whitening each
 * component of the reprojection error separately would result in rejecting individual components
 * as outliers. Instead, we minimize the whitened norm of the full reprojection error for each
 * point.  See
 * :meth:`ScalarNoiseModel.whiten_norm <symforce.opt.noise_models.ScalarNoiseModel.whiten_norm>`
 * for more information on this, and
 * :class:`BarronNoiseModel <symforce.opt.noise_models.BarronNoiseModel>` for more information on
 * the noise model.
 *
 * Args:
 *     source_pose: The pose of the source camera
 *     target_pose: The pose of the target camera
 *     target_calibration: The target spherical camera calibration
 *     source_inverse_range: The inverse range of the landmark in the source camera
 *     p_camera_source: The location of the landmark in the source camera coordinate, will be
 *         normalized
 *     target_pixel: The location of the correspondence in the target camera
 *     weight: The weight of the factor
 *     gnc_mu: The mu convexity parameter for the
 *         :class:`BarronNoiseModel <symforce.opt.noise_models.BarronNoiseModel>`
 *     gnc_scale: The scale parameter for the
 *         :class:`BarronNoiseModel <symforce.opt.noise_models.BarronNoiseModel>`
 *     epsilon: Small positive value
 *
 * Outputs:
 *     res: 2dof whiten residual of the reprojection
 *     jacobian: (2x13) jacobian of res wrt args source_pose (6), target_pose (6),
 *               source_inverse_range (1)
 *     hessian: (13x13) Gauss-Newton hessian for args source_pose (6), target_pose (6),
 *              source_inverse_range (1)
 *     rhs: (13x1) Gauss-Newton rhs for args source_pose (6), target_pose (6), source_inverse_range
 *          (1)
 */
template <typename Scalar>
void InverseRangeLandmarkPolynomialGncFactor(
    const sym::Pose3<Scalar>& source_pose, const sym::Pose3<Scalar>& target_pose,
    const sym::PolynomialCameraCal<Scalar>& target_calibration, const Scalar source_inverse_range,
    const Eigen::Matrix<Scalar, 3, 1>& p_camera_source,
    const Eigen::Matrix<Scalar, 2, 1>& target_pixel, const Scalar weight, const Scalar gnc_mu,
    const Scalar gnc_scale, const Scalar epsilon, Eigen::Matrix<Scalar, 2, 1>* const res = nullptr,
    Eigen::Matrix<Scalar, 2, 13>* const jacobian = nullptr,
    Eigen::Matrix<Scalar, 13, 13>* const hessian = nullptr,
    Eigen::Matrix<Scalar, 13, 1>* const rhs = nullptr) {
  // Total ops: 1126

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _source_pose = source_pose.Data();
  const Eigen::Matrix<Scalar, 7, 1>& _target_pose = target_pose.Data();
  const Eigen::Matrix<Scalar, 8, 1>& _target_calibration = target_calibration.Data();

  // Intermediate terms (301)
  const Scalar _tmp0 = std::pow(_target_pose[1], Scalar(2));
  const Scalar _tmp1 = -2 * _tmp0;
  const Scalar _tmp2 = std::pow(_target_pose[2], Scalar(2));
  const Scalar _tmp3 = 1 - 2 * _tmp2;
  const Scalar _tmp4 = _tmp1 + _tmp3;
  const Scalar _tmp5 = _source_pose[4] - _target_pose[4];
  const Scalar _tmp6 = 2 * _source_pose[0];
  const Scalar _tmp7 = _source_pose[1] * _tmp6;
  const Scalar _tmp8 = 2 * _source_pose[2];
  const Scalar _tmp9 = _source_pose[3] * _tmp8;
  const Scalar _tmp10 = -_tmp9;
  const Scalar _tmp11 = _tmp10 + _tmp7;
  const Scalar _tmp12 = std::pow(Scalar(epsilon + std::pow(p_camera_source(0, 0), Scalar(2)) +
                                        std::pow(p_camera_source(1, 0), Scalar(2)) +
                                        std::pow(p_camera_source(2, 0), Scalar(2))),
                                 Scalar(Scalar(-1) / Scalar(2)));
  const Scalar _tmp13 = _tmp12 * p_camera_source(1, 0);
  const Scalar _tmp14 = std::pow(_source_pose[2], Scalar(2));
  const Scalar _tmp15 = -2 * _tmp14;
  const Scalar _tmp16 = std::pow(_source_pose[1], Scalar(2));
  const Scalar _tmp17 = -2 * _tmp16;
  const Scalar _tmp18 = _tmp12 * p_camera_source(0, 0);
  const Scalar _tmp19 = _source_pose[0] * _tmp8;
  const Scalar _tmp20 = 2 * _source_pose[1] * _source_pose[3];
  const Scalar _tmp21 = _tmp19 + _tmp20;
  const Scalar _tmp22 = _tmp12 * p_camera_source(2, 0);
  const Scalar _tmp23 = _tmp11 * _tmp13 + _tmp18 * (_tmp15 + _tmp17 + 1) + _tmp21 * _tmp22 +
                        _tmp5 * source_inverse_range;
  const Scalar _tmp24 = 2 * _target_pose[2];
  const Scalar _tmp25 = _target_pose[3] * _tmp24;
  const Scalar _tmp26 = 2 * _target_pose[1];
  const Scalar _tmp27 = _target_pose[0] * _tmp26;
  const Scalar _tmp28 = _tmp25 + _tmp27;
  const Scalar _tmp29 = _source_pose[5] - _target_pose[5];
  const Scalar _tmp30 = _tmp7 + _tmp9;
  const Scalar _tmp31 = std::pow(_source_pose[0], Scalar(2));
  const Scalar _tmp32 = 1 - 2 * _tmp31;
  const Scalar _tmp33 = _source_pose[3] * _tmp6;
  const Scalar _tmp34 = -_tmp33;
  const Scalar _tmp35 = _source_pose[1] * _tmp8;
  const Scalar _tmp36 = _tmp34 + _tmp35;
  const Scalar _tmp37 = _tmp13 * (_tmp15 + _tmp32) + _tmp18 * _tmp30 + _tmp22 * _tmp36 +
                        _tmp29 * source_inverse_range;
  const Scalar _tmp38 = _target_pose[0] * _tmp24;
  const Scalar _tmp39 = _target_pose[3] * _tmp26;
  const Scalar _tmp40 = -_tmp39;
  const Scalar _tmp41 = _tmp38 + _tmp40;
  const Scalar _tmp42 = _source_pose[6] - _target_pose[6];
  const Scalar _tmp43 = -_tmp20;
  const Scalar _tmp44 = _tmp19 + _tmp43;
  const Scalar _tmp45 = _tmp33 + _tmp35;
  const Scalar _tmp46 = _tmp13 * _tmp45 + _tmp18 * _tmp44 + _tmp22 * (_tmp17 + _tmp32) +
                        _tmp42 * source_inverse_range;
  const Scalar _tmp47 = _tmp28 * _tmp37 + _tmp41 * _tmp46;
  const Scalar _tmp48 = _tmp23 * _tmp4 + _tmp47;
  const Scalar _tmp49 = std::pow(_target_pose[0], Scalar(2));
  const Scalar _tmp50 = -2 * _tmp49;
  const Scalar _tmp51 = _tmp3 + _tmp50;
  const Scalar _tmp52 = _target_pose[2] * _tmp26;
  const Scalar _tmp53 = 2 * _target_pose[0] * _target_pose[3];
  const Scalar _tmp54 = _tmp52 + _tmp53;
  const Scalar _tmp55 = -_tmp25;
  const Scalar _tmp56 = _tmp27 + _tmp55;
  const Scalar _tmp57 = _tmp23 * _tmp56 + _tmp46 * _tmp54;
  const Scalar _tmp58 = _tmp37 * _tmp51 + _tmp57;
  const Scalar _tmp59 = std::pow(_tmp58, Scalar(2));
  const Scalar _tmp60 = _tmp1 + _tmp50 + 1;
  const Scalar _tmp61 = -_tmp53;
  const Scalar _tmp62 = _tmp52 + _tmp61;
  const Scalar _tmp63 = _tmp38 + _tmp39;
  const Scalar _tmp64 = _tmp23 * _tmp63 + _tmp37 * _tmp62;
  const Scalar _tmp65 = _tmp46 * _tmp60 + _tmp64;
  const Scalar _tmp66 = std::max<Scalar>(_tmp65, epsilon);
  const Scalar _tmp67 = std::pow(_tmp66, Scalar(-2));
  const Scalar _tmp68 = std::pow(_tmp48, Scalar(2));
  const Scalar _tmp69 = _tmp59 * _tmp67 + _tmp67 * _tmp68 + epsilon;
  const Scalar _tmp70 = std::pow(_tmp69, Scalar(2));
  const Scalar _tmp71 = Scalar(1.0) * _target_calibration[5];
  const Scalar _tmp72 = Scalar(1.0) * _target_calibration[6] * _tmp70 +
                        Scalar(1.0) * _target_calibration[7] *
                            [&]() {
                              const Scalar base = _tmp69;
                              return base * base * base;
                            }() +
                        _tmp69 * _tmp71 + Scalar(1.0);
  const Scalar _tmp73 = Scalar(1.0) / (_tmp66);
  const Scalar _tmp74 = _target_calibration[0] * _tmp73;
  const Scalar _tmp75 = _tmp72 * _tmp74;
  const Scalar _tmp76 = _target_calibration[2] + _tmp48 * _tmp75 - target_pixel(0, 0);
  const Scalar _tmp77 = _target_calibration[1] * _tmp73;
  const Scalar _tmp78 = _tmp58 * _tmp77;
  const Scalar _tmp79 = _target_calibration[3] + _tmp72 * _tmp78 - target_pixel(1, 0);
  const Scalar _tmp80 = std::pow(_tmp76, Scalar(2)) + std::pow(_tmp79, Scalar(2)) + epsilon;
  const Scalar _tmp81 = std::pow(_tmp80, Scalar(Scalar(-1) / Scalar(2)));
  const Scalar _tmp82 = std::sqrt(weight);
  const Scalar _tmp83 =
      std::max<Scalar>(0, std::min<Scalar>((((_tmp65) > 0) - ((_tmp65) < 0)),
                                           (((_target_calibration[4] - std::sqrt(_tmp69)) > 0) -
                                            ((_target_calibration[4] - std::sqrt(_tmp69)) < 0))));
  const Scalar _tmp84 = Scalar(1.0) / (epsilon - gnc_mu + 1);
  const Scalar _tmp85 = epsilon + std::fabs(_tmp84);
  const Scalar _tmp86 = 2 - _tmp84;
  const Scalar _tmp87 =
      _tmp86 + epsilon * (2 * std::min<Scalar>(0, (((_tmp86) > 0) - ((_tmp86) < 0))) + 1);
  const Scalar _tmp88 = std::pow(gnc_scale, Scalar(-2));
  const Scalar _tmp89 = _tmp80 * _tmp88 / _tmp85 + 1;
  const Scalar _tmp90 = (Scalar(1) / Scalar(2)) * _tmp87;
  const Scalar _tmp91 =
      std::sqrt(Scalar(2)) * std::sqrt(Scalar(_tmp85 * (std::pow(_tmp89, _tmp90) - 1) / _tmp87));
  const Scalar _tmp92 = _tmp82 * _tmp83 * _tmp91;
  const Scalar _tmp93 = _tmp81 * _tmp92;
  const Scalar _tmp94 = _tmp76 * _tmp93;
  const Scalar _tmp95 = _tmp79 * _tmp93;
  const Scalar _tmp96 = std::pow(_source_pose[3], Scalar(2));
  const Scalar _tmp97 = -_tmp96;
  const Scalar _tmp98 = -_tmp16;
  const Scalar _tmp99 = _tmp13 * _tmp36 + _tmp22 * (_tmp14 + _tmp31 + _tmp97 + _tmp98);
  const Scalar _tmp100 = -_tmp31;
  const Scalar _tmp101 = _tmp100 + _tmp14;
  const Scalar _tmp102 = _tmp96 + _tmp98;
  const Scalar _tmp103 = -_tmp35;
  const Scalar _tmp104 = _tmp13 * (_tmp101 + _tmp102) + _tmp22 * (_tmp103 + _tmp34);
  const Scalar _tmp105 = -_tmp7;
  const Scalar _tmp106 = _tmp13 * _tmp21 + _tmp22 * (_tmp105 + _tmp9);
  const Scalar _tmp107 = _tmp104 * _tmp41 + _tmp106 * _tmp4 + _tmp28 * _tmp99;
  const Scalar _tmp108 = 2 * _tmp67;
  const Scalar _tmp109 = _tmp108 * _tmp48;
  const Scalar _tmp110 = _tmp104 * _tmp60 + _tmp106 * _tmp63 + _tmp62 * _tmp99;
  const Scalar _tmp111 = (((_tmp65 - epsilon) > 0) - ((_tmp65 - epsilon) < 0)) + 1;
  const Scalar _tmp112 = _tmp111 / [&]() {
    const Scalar base = _tmp66;
    return base * base * base;
  }();
  const Scalar _tmp113 = _tmp112 * _tmp68;
  const Scalar _tmp114 = _tmp104 * _tmp54 + _tmp106 * _tmp56 + _tmp51 * _tmp99;
  const Scalar _tmp115 = _tmp108 * _tmp58;
  const Scalar _tmp116 = _tmp112 * _tmp59;
  const Scalar _tmp117 =
      _tmp107 * _tmp109 - _tmp110 * _tmp113 - _tmp110 * _tmp116 + _tmp114 * _tmp115;
  const Scalar _tmp118 = Scalar(2.0) * _target_calibration[6] * _tmp69;
  const Scalar _tmp119 = Scalar(3.0) * _target_calibration[7] * _tmp70;
  const Scalar _tmp120 = _tmp117 * _tmp118 + _tmp117 * _tmp119 + _tmp117 * _tmp71;
  const Scalar _tmp121 = _tmp48 * _tmp74;
  const Scalar _tmp122 = (Scalar(1) / Scalar(2)) * _tmp111 * _tmp67 * _tmp72;
  const Scalar _tmp123 = _tmp110 * _tmp122;
  const Scalar _tmp124 = _target_calibration[0] * _tmp48;
  const Scalar _tmp125 = _tmp107 * _tmp75 + _tmp120 * _tmp121 - _tmp123 * _tmp124;
  const Scalar _tmp126 = _target_calibration[1] * _tmp58;
  const Scalar _tmp127 = _tmp72 * _tmp77;
  const Scalar _tmp128 = _tmp114 * _tmp127 + _tmp120 * _tmp78 - _tmp123 * _tmp126;
  const Scalar _tmp129 = 2 * _tmp79;
  const Scalar _tmp130 = 2 * _tmp76;
  const Scalar _tmp131 = _tmp125 * _tmp130 + _tmp128 * _tmp129;
  const Scalar _tmp132 = _tmp92 / (_tmp80 * std::sqrt(_tmp80));
  const Scalar _tmp133 = (Scalar(1) / Scalar(2)) * _tmp76;
  const Scalar _tmp134 = _tmp132 * _tmp133;
  const Scalar _tmp135 =
      _tmp81 * _tmp82 * _tmp83 * _tmp88 * std::pow(_tmp89, Scalar(_tmp90 - 1)) / _tmp91;
  const Scalar _tmp136 = _tmp133 * _tmp135;
  const Scalar _tmp137 = _tmp125 * _tmp93 - _tmp131 * _tmp134 + _tmp131 * _tmp136;
  const Scalar _tmp138 = (Scalar(1) / Scalar(2)) * _tmp79;
  const Scalar _tmp139 = _tmp135 * _tmp138;
  const Scalar _tmp140 = _tmp132 * _tmp138;
  const Scalar _tmp141 = _tmp128 * _tmp93 + _tmp131 * _tmp139 - _tmp131 * _tmp140;
  const Scalar _tmp142 = _tmp18 * (_tmp103 + _tmp33) + _tmp22 * _tmp30;
  const Scalar _tmp143 = _tmp16 + _tmp97;
  const Scalar _tmp144 = -_tmp14;
  const Scalar _tmp145 = _tmp144 + _tmp31;
  const Scalar _tmp146 = _tmp18 * (_tmp143 + _tmp145) + _tmp22 * _tmp44;
  const Scalar _tmp147 = -_tmp19;
  const Scalar _tmp148 = _tmp18 * (_tmp147 + _tmp43) + _tmp22 * (_tmp102 + _tmp145);
  const Scalar _tmp149 = _tmp142 * _tmp62 + _tmp146 * _tmp60 + _tmp148 * _tmp63;
  const Scalar _tmp150 = _tmp142 * _tmp51 + _tmp146 * _tmp54 + _tmp148 * _tmp56;
  const Scalar _tmp151 = _tmp142 * _tmp28 + _tmp146 * _tmp41 + _tmp148 * _tmp4;
  const Scalar _tmp152 =
      _tmp109 * _tmp151 - _tmp113 * _tmp149 + _tmp115 * _tmp150 - _tmp116 * _tmp149;
  const Scalar _tmp153 = _tmp118 * _tmp152 + _tmp119 * _tmp152 + _tmp152 * _tmp71;
  const Scalar _tmp154 = _tmp122 * _tmp126;
  const Scalar _tmp155 = _tmp127 * _tmp150 - _tmp149 * _tmp154 + _tmp153 * _tmp78;
  const Scalar _tmp156 = _tmp122 * _tmp124;
  const Scalar _tmp157 = _tmp121 * _tmp153 - _tmp149 * _tmp156 + _tmp151 * _tmp75;
  const Scalar _tmp158 = _tmp129 * _tmp155 + _tmp130 * _tmp157;
  const Scalar _tmp159 = -_tmp134 * _tmp158 + _tmp136 * _tmp158 + _tmp157 * _tmp93;
  const Scalar _tmp160 = _tmp139 * _tmp158 - _tmp140 * _tmp158 + _tmp155 * _tmp93;
  const Scalar _tmp161 = _tmp13 * (_tmp147 + _tmp20) + _tmp18 * _tmp45;
  const Scalar _tmp162 =
      _tmp13 * (_tmp10 + _tmp105) + _tmp18 * (_tmp100 + _tmp144 + _tmp16 + _tmp96);
  const Scalar _tmp163 = _tmp11 * _tmp18 + _tmp13 * (_tmp101 + _tmp143);
  const Scalar _tmp164 = _tmp161 * _tmp41 + _tmp162 * _tmp28 + _tmp163 * _tmp4;
  const Scalar _tmp165 = _tmp161 * _tmp54 + _tmp162 * _tmp51 + _tmp163 * _tmp56;
  const Scalar _tmp166 = _tmp161 * _tmp60 + _tmp162 * _tmp62 + _tmp163 * _tmp63;
  const Scalar _tmp167 =
      _tmp109 * _tmp164 - _tmp113 * _tmp166 + _tmp115 * _tmp165 - _tmp116 * _tmp166;
  const Scalar _tmp168 = _tmp118 * _tmp167 + _tmp119 * _tmp167 + _tmp167 * _tmp71;
  const Scalar _tmp169 = _tmp121 * _tmp168 - _tmp156 * _tmp166 + _tmp164 * _tmp75;
  const Scalar _tmp170 = _tmp127 * _tmp165 - _tmp154 * _tmp166 + _tmp168 * _tmp78;
  const Scalar _tmp171 = _tmp129 * _tmp170 + _tmp130 * _tmp169;
  const Scalar _tmp172 = -_tmp134 * _tmp171 + _tmp136 * _tmp171 + _tmp169 * _tmp93;
  const Scalar _tmp173 = _tmp139 * _tmp171 - _tmp140 * _tmp171 + _tmp170 * _tmp93;
  const Scalar _tmp174 = _tmp154 * source_inverse_range;
  const Scalar _tmp175 = _tmp174 * _tmp63;
  const Scalar _tmp176 = _tmp127 * source_inverse_range;
  const Scalar _tmp177 = _tmp176 * _tmp56;
  const Scalar _tmp178 = _tmp115 * source_inverse_range;
  const Scalar _tmp179 = _tmp178 * _tmp56;
  const Scalar _tmp180 = _tmp109 * source_inverse_range;
  const Scalar _tmp181 = _tmp180 * _tmp4;
  const Scalar _tmp182 = _tmp113 * source_inverse_range;
  const Scalar _tmp183 = _tmp182 * _tmp63;
  const Scalar _tmp184 = _tmp116 * source_inverse_range;
  const Scalar _tmp185 = _tmp184 * _tmp63;
  const Scalar _tmp186 = _tmp179 + _tmp181 - _tmp183 - _tmp185;
  const Scalar _tmp187 = _tmp118 * _tmp186 + _tmp119 * _tmp186 + _tmp186 * _tmp71;
  const Scalar _tmp188 = -_tmp175 + _tmp177 + _tmp187 * _tmp78;
  const Scalar _tmp189 = _tmp75 * source_inverse_range;
  const Scalar _tmp190 = _tmp189 * _tmp4;
  const Scalar _tmp191 = _tmp156 * source_inverse_range;
  const Scalar _tmp192 = _tmp191 * _tmp63;
  const Scalar _tmp193 = _tmp121 * _tmp187 + _tmp190 - _tmp192;
  const Scalar _tmp194 = _tmp129 * _tmp188 + _tmp130 * _tmp193;
  const Scalar _tmp195 = _tmp132 * _tmp194;
  const Scalar _tmp196 = -_tmp133 * _tmp195 + _tmp136 * _tmp194 + _tmp193 * _tmp93;
  const Scalar _tmp197 = -_tmp138 * _tmp195 + _tmp139 * _tmp194 + _tmp188 * _tmp93;
  const Scalar _tmp198 = _tmp174 * _tmp62;
  const Scalar _tmp199 = _tmp176 * _tmp51;
  const Scalar _tmp200 = _tmp178 * _tmp51;
  const Scalar _tmp201 = _tmp182 * _tmp62;
  const Scalar _tmp202 = _tmp184 * _tmp62;
  const Scalar _tmp203 = _tmp180 * _tmp28;
  const Scalar _tmp204 = _tmp200 - _tmp201 - _tmp202 + _tmp203;
  const Scalar _tmp205 = _tmp118 * _tmp204 + _tmp119 * _tmp204 + _tmp204 * _tmp71;
  const Scalar _tmp206 = -_tmp198 + _tmp199 + _tmp205 * _tmp78;
  const Scalar _tmp207 = _tmp189 * _tmp28;
  const Scalar _tmp208 = _tmp191 * _tmp62;
  const Scalar _tmp209 = _tmp121 * _tmp205 + _tmp207 - _tmp208;
  const Scalar _tmp210 = _tmp129 * _tmp206 + _tmp130 * _tmp209;
  const Scalar _tmp211 = -_tmp134 * _tmp210 + _tmp136 * _tmp210 + _tmp209 * _tmp93;
  const Scalar _tmp212 = _tmp139 * _tmp210 - _tmp140 * _tmp210 + _tmp206 * _tmp93;
  const Scalar _tmp213 = _tmp182 * _tmp60;
  const Scalar _tmp214 = _tmp180 * _tmp41;
  const Scalar _tmp215 = _tmp184 * _tmp60;
  const Scalar _tmp216 = _tmp54 * source_inverse_range;
  const Scalar _tmp217 = _tmp115 * _tmp216;
  const Scalar _tmp218 = -_tmp213 + _tmp214 - _tmp215 + _tmp217;
  const Scalar _tmp219 = _tmp118 * _tmp218 + _tmp119 * _tmp218 + _tmp218 * _tmp71;
  const Scalar _tmp220 = _tmp127 * _tmp216;
  const Scalar _tmp221 = _tmp174 * _tmp60;
  const Scalar _tmp222 = _tmp219 * _tmp78 + _tmp220 - _tmp221;
  const Scalar _tmp223 = _tmp191 * _tmp60;
  const Scalar _tmp224 = _tmp189 * _tmp41;
  const Scalar _tmp225 = _tmp121 * _tmp219 - _tmp223 + _tmp224;
  const Scalar _tmp226 = _tmp129 * _tmp222 + _tmp130 * _tmp225;
  const Scalar _tmp227 = -_tmp134 * _tmp226 + _tmp136 * _tmp226 + _tmp225 * _tmp93;
  const Scalar _tmp228 = _tmp138 * _tmp226;
  const Scalar _tmp229 = -_tmp132 * _tmp228 + _tmp135 * _tmp228 + _tmp222 * _tmp93;
  const Scalar _tmp230 = std::pow(_target_pose[3], Scalar(2));
  const Scalar _tmp231 = -_tmp230;
  const Scalar _tmp232 = _tmp231 + _tmp49;
  const Scalar _tmp233 = -_tmp0;
  const Scalar _tmp234 = _tmp2 + _tmp233;
  const Scalar _tmp235 = -_tmp52;
  const Scalar _tmp236 = -_tmp27;
  const Scalar _tmp237 =
      _tmp23 * (_tmp236 + _tmp25) + _tmp37 * (_tmp232 + _tmp234) + _tmp46 * (_tmp235 + _tmp61);
  const Scalar _tmp238 = -_tmp49;
  const Scalar _tmp239 = _tmp230 + _tmp238;
  const Scalar _tmp240 = _tmp46 * (_tmp234 + _tmp239) + _tmp64;
  const Scalar _tmp241 = -_tmp113 * _tmp237 + _tmp115 * _tmp240 - _tmp116 * _tmp237;
  const Scalar _tmp242 = _tmp118 * _tmp241 + _tmp119 * _tmp241 + _tmp241 * _tmp71;
  const Scalar _tmp243 = _tmp121 * _tmp242 - _tmp156 * _tmp237;
  const Scalar _tmp244 = _tmp127 * _tmp240 - _tmp154 * _tmp237 + _tmp242 * _tmp78;
  const Scalar _tmp245 = _tmp129 * _tmp244 + _tmp130 * _tmp243;
  const Scalar _tmp246 = -_tmp134 * _tmp245 + _tmp136 * _tmp245 + _tmp243 * _tmp93;
  const Scalar _tmp247 = _tmp139 * _tmp245 - _tmp140 * _tmp245 + _tmp244 * _tmp93;
  const Scalar _tmp248 = -_tmp2;
  const Scalar _tmp249 = _tmp0 + _tmp248;
  const Scalar _tmp250 = -_tmp38;
  const Scalar _tmp251 =
      _tmp23 * (_tmp250 + _tmp40) + _tmp37 * (_tmp235 + _tmp53) + _tmp46 * (_tmp232 + _tmp249);
  const Scalar _tmp252 = _tmp23 * (_tmp230 + _tmp233 + _tmp248 + _tmp49) + _tmp47;
  const Scalar _tmp253 = _tmp109 * _tmp251 - _tmp113 * _tmp252 - _tmp116 * _tmp252;
  const Scalar _tmp254 = _tmp118 * _tmp253 + _tmp119 * _tmp253 + _tmp253 * _tmp71;
  const Scalar _tmp255 = _tmp121 * _tmp254 - _tmp156 * _tmp252 + _tmp251 * _tmp75;
  const Scalar _tmp256 = -_tmp154 * _tmp252 + _tmp254 * _tmp78;
  const Scalar _tmp257 = _tmp129 * _tmp256 + _tmp130 * _tmp255;
  const Scalar _tmp258 = -_tmp134 * _tmp257 + _tmp136 * _tmp257 + _tmp255 * _tmp93;
  const Scalar _tmp259 = _tmp139 * _tmp257 - _tmp140 * _tmp257 + _tmp256 * _tmp93;
  const Scalar _tmp260 = _tmp37 * (_tmp239 + _tmp249) + _tmp57;
  const Scalar _tmp261 = _tmp23 * (_tmp0 + _tmp2 + _tmp231 + _tmp238) +
                         _tmp37 * (_tmp236 + _tmp55) + _tmp46 * (_tmp250 + _tmp39);
  const Scalar _tmp262 = _tmp109 * _tmp260 + _tmp115 * _tmp261;
  const Scalar _tmp263 = _tmp118 * _tmp262 + _tmp119 * _tmp262 + _tmp262 * _tmp71;
  const Scalar _tmp264 = _tmp121 * _tmp263 + _tmp260 * _tmp75;
  const Scalar _tmp265 = _tmp127 * _tmp261 + _tmp263 * _tmp78;
  const Scalar _tmp266 = _tmp129 * _tmp265 + _tmp130 * _tmp264;
  const Scalar _tmp267 = -_tmp134 * _tmp266 + _tmp136 * _tmp266 + _tmp264 * _tmp93;
  const Scalar _tmp268 = _tmp139 * _tmp266 - _tmp140 * _tmp266 + _tmp265 * _tmp93;
  const Scalar _tmp269 = -_tmp179 - _tmp181 + _tmp183 + _tmp185;
  const Scalar _tmp270 = _tmp118 * _tmp269 + _tmp119 * _tmp269 + _tmp269 * _tmp71;
  const Scalar _tmp271 = _tmp175 - _tmp177 + _tmp270 * _tmp78;
  const Scalar _tmp272 = _tmp121 * _tmp270 - _tmp190 + _tmp192;
  const Scalar _tmp273 = _tmp129 * _tmp271 + _tmp130 * _tmp272;
  const Scalar _tmp274 = -_tmp134 * _tmp273 + _tmp136 * _tmp273 + _tmp272 * _tmp93;
  const Scalar _tmp275 = _tmp139 * _tmp273 - _tmp140 * _tmp273 + _tmp271 * _tmp93;
  const Scalar _tmp276 = -_tmp200 + _tmp201 + _tmp202 - _tmp203;
  const Scalar _tmp277 = _tmp118 * _tmp276 + _tmp119 * _tmp276 + _tmp276 * _tmp71;
  const Scalar _tmp278 = _tmp121 * _tmp277 - _tmp207 + _tmp208;
  const Scalar _tmp279 = _tmp198 - _tmp199 + _tmp277 * _tmp78;
  const Scalar _tmp280 = _tmp129 * _tmp279 + _tmp130 * _tmp278;
  const Scalar _tmp281 = -_tmp134 * _tmp280 + _tmp136 * _tmp280 + _tmp278 * _tmp93;
  const Scalar _tmp282 = _tmp139 * _tmp280 - _tmp140 * _tmp280 + _tmp279 * _tmp93;
  const Scalar _tmp283 = _tmp213 - _tmp214 + _tmp215 - _tmp217;
  const Scalar _tmp284 = _tmp118 * _tmp283 + _tmp119 * _tmp283 + _tmp283 * _tmp71;
  const Scalar _tmp285 = -_tmp220 + _tmp221 + _tmp284 * _tmp78;
  const Scalar _tmp286 = _tmp121 * _tmp284 + _tmp223 - _tmp224;
  const Scalar _tmp287 = _tmp129 * _tmp285 + _tmp130 * _tmp286;
  const Scalar _tmp288 = -_tmp134 * _tmp287 + _tmp136 * _tmp287 + _tmp286 * _tmp93;
  const Scalar _tmp289 = _tmp139 * _tmp287 - _tmp140 * _tmp287 + _tmp285 * _tmp93;
  const Scalar _tmp290 = _tmp28 * _tmp29 + _tmp4 * _tmp5 + _tmp41 * _tmp42;
  const Scalar _tmp291 = _tmp29 * _tmp62 + _tmp42 * _tmp60 + _tmp5 * _tmp63;
  const Scalar _tmp292 = _tmp29 * _tmp51 + _tmp42 * _tmp54 + _tmp5 * _tmp56;
  const Scalar _tmp293 =
      _tmp109 * _tmp290 - _tmp113 * _tmp291 + _tmp115 * _tmp292 - _tmp116 * _tmp291;
  const Scalar _tmp294 = _tmp118 * _tmp293 + _tmp119 * _tmp293 + _tmp293 * _tmp71;
  const Scalar _tmp295 = _tmp122 * _tmp291;
  const Scalar _tmp296 = _tmp121 * _tmp294 - _tmp124 * _tmp295 + _tmp290 * _tmp75;
  const Scalar _tmp297 = -_tmp126 * _tmp295 + _tmp127 * _tmp292 + _tmp294 * _tmp78;
  const Scalar _tmp298 = _tmp129 * _tmp297 + _tmp130 * _tmp296;
  const Scalar _tmp299 = -_tmp134 * _tmp298 + _tmp136 * _tmp298 + _tmp296 * _tmp93;
  const Scalar _tmp300 = _tmp139 * _tmp298 - _tmp140 * _tmp298 + _tmp297 * _tmp93;

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 2, 1>& _res = (*res);

    _res(0, 0) = _tmp94;
    _res(1, 0) = _tmp95;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 2, 13>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp137;
    _jacobian(1, 0) = _tmp141;
    _jacobian(0, 1) = _tmp159;
    _jacobian(1, 1) = _tmp160;
    _jacobian(0, 2) = _tmp172;
    _jacobian(1, 2) = _tmp173;
    _jacobian(0, 3) = _tmp196;
    _jacobian(1, 3) = _tmp197;
    _jacobian(0, 4) = _tmp211;
    _jacobian(1, 4) = _tmp212;
    _jacobian(0, 5) = _tmp227;
    _jacobian(1, 5) = _tmp229;
    _jacobian(0, 6) = _tmp246;
    _jacobian(1, 6) = _tmp247;
    _jacobian(0, 7) = _tmp258;
    _jacobian(1, 7) = _tmp259;
    _jacobian(0, 8) = _tmp267;
    _jacobian(1, 8) = _tmp268;
    _jacobian(0, 9) = _tmp274;
    _jacobian(1, 9) = _tmp275;
    _jacobian(0, 10) = _tmp281;
    _jacobian(1, 10) = _tmp282;
    _jacobian(0, 11) = _tmp288;
    _jacobian(1, 11) = _tmp289;
    _jacobian(0, 12) = _tmp299;
    _jacobian(1, 12) = _tmp300;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 13, 13>& _hessian = (*hessian);

    _hessian(0, 0) = std::pow(_tmp137, Scalar(2)) + std::pow(_tmp141, Scalar(2));
    _hessian(1, 0) = _tmp137 * _tmp159 + _tmp141 * _tmp160;
    _hessian(2, 0) = _tmp137 * _tmp172 + _tmp141 * _tmp173;
    _hessian(3, 0) = _tmp137 * _tmp196 + _tmp141 * _tmp197;
    _hessian(4, 0) = _tmp137 * _tmp211 + _tmp141 * _tmp212;
    _hessian(5, 0) = _tmp137 * _tmp227 + _tmp141 * _tmp229;
    _hessian(6, 0) = _tmp137 * _tmp246 + _tmp141 * _tmp247;
    _hessian(7, 0) = _tmp137 * _tmp258 + _tmp141 * _tmp259;
    _hessian(8, 0) = _tmp137 * _tmp267 + _tmp141 * _tmp268;
    _hessian(9, 0) = _tmp137 * _tmp274 + _tmp141 * _tmp275;
    _hessian(10, 0) = _tmp137 * _tmp281 + _tmp141 * _tmp282;
    _hessian(11, 0) = _tmp137 * _tmp288 + _tmp141 * _tmp289;
    _hessian(12, 0) = _tmp137 * _tmp299 + _tmp141 * _tmp300;
    _hessian(0, 1) = 0;
    _hessian(1, 1) = std::pow(_tmp159, Scalar(2)) + std::pow(_tmp160, Scalar(2));
    _hessian(2, 1) = _tmp159 * _tmp172 + _tmp160 * _tmp173;
    _hessian(3, 1) = _tmp159 * _tmp196 + _tmp160 * _tmp197;
    _hessian(4, 1) = _tmp159 * _tmp211 + _tmp160 * _tmp212;
    _hessian(5, 1) = _tmp159 * _tmp227 + _tmp160 * _tmp229;
    _hessian(6, 1) = _tmp159 * _tmp246 + _tmp160 * _tmp247;
    _hessian(7, 1) = _tmp159 * _tmp258 + _tmp160 * _tmp259;
    _hessian(8, 1) = _tmp159 * _tmp267 + _tmp160 * _tmp268;
    _hessian(9, 1) = _tmp159 * _tmp274 + _tmp160 * _tmp275;
    _hessian(10, 1) = _tmp159 * _tmp281 + _tmp160 * _tmp282;
    _hessian(11, 1) = _tmp159 * _tmp288 + _tmp160 * _tmp289;
    _hessian(12, 1) = _tmp159 * _tmp299 + _tmp160 * _tmp300;
    _hessian(0, 2) = 0;
    _hessian(1, 2) = 0;
    _hessian(2, 2) = std::pow(_tmp172, Scalar(2)) + std::pow(_tmp173, Scalar(2));
    _hessian(3, 2) = _tmp172 * _tmp196 + _tmp173 * _tmp197;
    _hessian(4, 2) = _tmp172 * _tmp211 + _tmp173 * _tmp212;
    _hessian(5, 2) = _tmp172 * _tmp227 + _tmp173 * _tmp229;
    _hessian(6, 2) = _tmp172 * _tmp246 + _tmp173 * _tmp247;
    _hessian(7, 2) = _tmp172 * _tmp258 + _tmp173 * _tmp259;
    _hessian(8, 2) = _tmp172 * _tmp267 + _tmp173 * _tmp268;
    _hessian(9, 2) = _tmp172 * _tmp274 + _tmp173 * _tmp275;
    _hessian(10, 2) = _tmp172 * _tmp281 + _tmp173 * _tmp282;
    _hessian(11, 2) = _tmp172 * _tmp288 + _tmp173 * _tmp289;
    _hessian(12, 2) = _tmp172 * _tmp299 + _tmp173 * _tmp300;
    _hessian(0, 3) = 0;
    _hessian(1, 3) = 0;
    _hessian(2, 3) = 0;
    _hessian(3, 3) = std::pow(_tmp196, Scalar(2)) + std::pow(_tmp197, Scalar(2));
    _hessian(4, 3) = _tmp196 * _tmp211 + _tmp197 * _tmp212;
    _hessian(5, 3) = _tmp196 * _tmp227 + _tmp197 * _tmp229;
    _hessian(6, 3) = _tmp196 * _tmp246 + _tmp197 * _tmp247;
    _hessian(7, 3) = _tmp196 * _tmp258 + _tmp197 * _tmp259;
    _hessian(8, 3) = _tmp196 * _tmp267 + _tmp197 * _tmp268;
    _hessian(9, 3) = _tmp196 * _tmp274 + _tmp197 * _tmp275;
    _hessian(10, 3) = _tmp196 * _tmp281 + _tmp197 * _tmp282;
    _hessian(11, 3) = _tmp196 * _tmp288 + _tmp197 * _tmp289;
    _hessian(12, 3) = _tmp196 * _tmp299 + _tmp197 * _tmp300;
    _hessian(0, 4) = 0;
    _hessian(1, 4) = 0;
    _hessian(2, 4) = 0;
    _hessian(3, 4) = 0;
    _hessian(4, 4) = std::pow(_tmp211, Scalar(2)) + std::pow(_tmp212, Scalar(2));
    _hessian(5, 4) = _tmp211 * _tmp227 + _tmp212 * _tmp229;
    _hessian(6, 4) = _tmp211 * _tmp246 + _tmp212 * _tmp247;
    _hessian(7, 4) = _tmp211 * _tmp258 + _tmp212 * _tmp259;
    _hessian(8, 4) = _tmp211 * _tmp267 + _tmp212 * _tmp268;
    _hessian(9, 4) = _tmp211 * _tmp274 + _tmp212 * _tmp275;
    _hessian(10, 4) = _tmp211 * _tmp281 + _tmp212 * _tmp282;
    _hessian(11, 4) = _tmp211 * _tmp288 + _tmp212 * _tmp289;
    _hessian(12, 4) = _tmp211 * _tmp299 + _tmp212 * _tmp300;
    _hessian(0, 5) = 0;
    _hessian(1, 5) = 0;
    _hessian(2, 5) = 0;
    _hessian(3, 5) = 0;
    _hessian(4, 5) = 0;
    _hessian(5, 5) = std::pow(_tmp227, Scalar(2)) + std::pow(_tmp229, Scalar(2));
    _hessian(6, 5) = _tmp227 * _tmp246 + _tmp229 * _tmp247;
    _hessian(7, 5) = _tmp227 * _tmp258 + _tmp229 * _tmp259;
    _hessian(8, 5) = _tmp227 * _tmp267 + _tmp229 * _tmp268;
    _hessian(9, 5) = _tmp227 * _tmp274 + _tmp229 * _tmp275;
    _hessian(10, 5) = _tmp227 * _tmp281 + _tmp229 * _tmp282;
    _hessian(11, 5) = _tmp227 * _tmp288 + _tmp229 * _tmp289;
    _hessian(12, 5) = _tmp227 * _tmp299 + _tmp229 * _tmp300;
    _hessian(0, 6) = 0;
    _hessian(1, 6) = 0;
    _hessian(2, 6) = 0;
    _hessian(3, 6) = 0;
    _hessian(4, 6) = 0;
    _hessian(5, 6) = 0;
    _hessian(6, 6) = std::pow(_tmp246, Scalar(2)) + std::pow(_tmp247, Scalar(2));
    _hessian(7, 6) = _tmp246 * _tmp258 + _tmp247 * _tmp259;
    _hessian(8, 6) = _tmp246 * _tmp267 + _tmp247 * _tmp268;
    _hessian(9, 6) = _tmp246 * _tmp274 + _tmp247 * _tmp275;
    _hessian(10, 6) = _tmp246 * _tmp281 + _tmp247 * _tmp282;
    _hessian(11, 6) = _tmp246 * _tmp288 + _tmp247 * _tmp289;
    _hessian(12, 6) = _tmp246 * _tmp299 + _tmp247 * _tmp300;
    _hessian(0, 7) = 0;
    _hessian(1, 7) = 0;
    _hessian(2, 7) = 0;
    _hessian(3, 7) = 0;
    _hessian(4, 7) = 0;
    _hessian(5, 7) = 0;
    _hessian(6, 7) = 0;
    _hessian(7, 7) = std::pow(_tmp258, Scalar(2)) + std::pow(_tmp259, Scalar(2));
    _hessian(8, 7) = _tmp258 * _tmp267 + _tmp259 * _tmp268;
    _hessian(9, 7) = _tmp258 * _tmp274 + _tmp259 * _tmp275;
    _hessian(10, 7) = _tmp258 * _tmp281 + _tmp259 * _tmp282;
    _hessian(11, 7) = _tmp258 * _tmp288 + _tmp259 * _tmp289;
    _hessian(12, 7) = _tmp258 * _tmp299 + _tmp259 * _tmp300;
    _hessian(0, 8) = 0;
    _hessian(1, 8) = 0;
    _hessian(2, 8) = 0;
    _hessian(3, 8) = 0;
    _hessian(4, 8) = 0;
    _hessian(5, 8) = 0;
    _hessian(6, 8) = 0;
    _hessian(7, 8) = 0;
    _hessian(8, 8) = std::pow(_tmp267, Scalar(2)) + std::pow(_tmp268, Scalar(2));
    _hessian(9, 8) = _tmp267 * _tmp274 + _tmp268 * _tmp275;
    _hessian(10, 8) = _tmp267 * _tmp281 + _tmp268 * _tmp282;
    _hessian(11, 8) = _tmp267 * _tmp288 + _tmp268 * _tmp289;
    _hessian(12, 8) = _tmp267 * _tmp299 + _tmp268 * _tmp300;
    _hessian(0, 9) = 0;
    _hessian(1, 9) = 0;
    _hessian(2, 9) = 0;
    _hessian(3, 9) = 0;
    _hessian(4, 9) = 0;
    _hessian(5, 9) = 0;
    _hessian(6, 9) = 0;
    _hessian(7, 9) = 0;
    _hessian(8, 9) = 0;
    _hessian(9, 9) = std::pow(_tmp274, Scalar(2)) + std::pow(_tmp275, Scalar(2));
    _hessian(10, 9) = _tmp274 * _tmp281 + _tmp275 * _tmp282;
    _hessian(11, 9) = _tmp274 * _tmp288 + _tmp275 * _tmp289;
    _hessian(12, 9) = _tmp274 * _tmp299 + _tmp275 * _tmp300;
    _hessian(0, 10) = 0;
    _hessian(1, 10) = 0;
    _hessian(2, 10) = 0;
    _hessian(3, 10) = 0;
    _hessian(4, 10) = 0;
    _hessian(5, 10) = 0;
    _hessian(6, 10) = 0;
    _hessian(7, 10) = 0;
    _hessian(8, 10) = 0;
    _hessian(9, 10) = 0;
    _hessian(10, 10) = std::pow(_tmp281, Scalar(2)) + std::pow(_tmp282, Scalar(2));
    _hessian(11, 10) = _tmp281 * _tmp288 + _tmp282 * _tmp289;
    _hessian(12, 10) = _tmp281 * _tmp299 + _tmp282 * _tmp300;
    _hessian(0, 11) = 0;
    _hessian(1, 11) = 0;
    _hessian(2, 11) = 0;
    _hessian(3, 11) = 0;
    _hessian(4, 11) = 0;
    _hessian(5, 11) = 0;
    _hessian(6, 11) = 0;
    _hessian(7, 11) = 0;
    _hessian(8, 11) = 0;
    _hessian(9, 11) = 0;
    _hessian(10, 11) = 0;
    _hessian(11, 11) = std::pow(_tmp288, Scalar(2)) + std::pow(_tmp289, Scalar(2));
    _hessian(12, 11) = _tmp288 * _tmp299 + _tmp289 * _tmp300;
    _hessian(0, 12) = 0;
    _hessian(1, 12) = 0;
    _hessian(2, 12) = 0;
    _hessian(3, 12) = 0;
    _hessian(4, 12) = 0;
    _hessian(5, 12) = 0;
    _hessian(6, 12) = 0;
    _hessian(7, 12) = 0;
    _hessian(8, 12) = 0;
    _hessian(9, 12) = 0;
    _hessian(10, 12) = 0;
    _hessian(11, 12) = 0;
    _hessian(12, 12) = std::pow(_tmp299, Scalar(2)) + std::pow(_tmp300, Scalar(2));
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 13, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp137 * _tmp94 + _tmp141 * _tmp95;
    _rhs(1, 0) = _tmp159 * _tmp94 + _tmp160 * _tmp95;
    _rhs(2, 0) = _tmp172 * _tmp94 + _tmp173 * _tmp95;
    _rhs(3, 0) = _tmp196 * _tmp94 + _tmp197 * _tmp95;
    _rhs(4, 0) = _tmp211 * _tmp94 + _tmp212 * _tmp95;
    _rhs(5, 0) = _tmp227 * _tmp94 + _tmp229 * _tmp95;
    _rhs(6, 0) = _tmp246 * _tmp94 + _tmp247 * _tmp95;
    _rhs(7, 0) = _tmp258 * _tmp94 + _tmp259 * _tmp95;
    _rhs(8, 0) = _tmp267 * _tmp94 + _tmp268 * _tmp95;
    _rhs(9, 0) = _tmp274 * _tmp94 + _tmp275 * _tmp95;
    _rhs(10, 0) = _tmp281 * _tmp94 + _tmp282 * _tmp95;
    _rhs(11, 0) = _tmp288 * _tmp94 + _tmp289 * _tmp95;
    _rhs(12, 0) = _tmp299 * _tmp94 + _tmp300 * _tmp95;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
