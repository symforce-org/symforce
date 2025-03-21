// -----------------------------------------------------------------------------
// This file was autogenerated by symforce from template:
//     function/FUNCTION.h.jinja
// Do NOT modify by hand.
// -----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

#include <sym/pose3.h>

namespace sym {

/**
 * Reprojection residual for the camera model used in the Bundle-Adjustment-in-the-Large dataset, a
 * polynomial camera with two distortion coefficients, cx == cy == 0, and fx == fy
 *
 * See https://grail.cs.washington.edu/projects/bal/ for more information
 *
 * Args:
 *     cam_T_world: The (inverse) pose of the camera
 *     intrinsics: Camera intrinsics (f, k1, k2)
 *     point: The world point to be projected
 *     pixel: The measured pixel in the camera (with (0, 0) == center of image)
 *
 * Returns:
 *     residual: The reprojection residual
 *     jacobian: (2x12) jacobian of res wrt args cam_T_world (6), intrinsics (3), point (3)
 *     hessian: (12x12) Gauss-Newton hessian for args cam_T_world (6), intrinsics (3), point (3)
 *     rhs: (12x1) Gauss-Newton rhs for args cam_T_world (6), intrinsics (3), point (3)
 */
template <typename Scalar>
void SnavelyReprojectionFactor(const sym::Pose3<Scalar>& cam_T_world,
                               const Eigen::Matrix<Scalar, 3, 1>& intrinsics,
                               const Eigen::Matrix<Scalar, 3, 1>& point,
                               const Eigen::Matrix<Scalar, 2, 1>& pixel, const Scalar epsilon,
                               Eigen::Matrix<Scalar, 2, 1>* const res = nullptr,
                               Eigen::Matrix<Scalar, 2, 12>* const jacobian = nullptr,
                               Eigen::Matrix<Scalar, 12, 12>* const hessian = nullptr,
                               Eigen::Matrix<Scalar, 12, 1>* const rhs = nullptr) {
  // Total ops: 688

  // Unused inputs
  (void)epsilon;

  // Input arrays
  const Eigen::Matrix<Scalar, 7, 1>& _cam_T_world = cam_T_world.Data();

  // Intermediate terms (175)
  const Scalar _tmp0 = 2 * _cam_T_world[3];
  const Scalar _tmp1 = _cam_T_world[2] * _tmp0;
  const Scalar _tmp2 = _cam_T_world[0] * _cam_T_world[1];
  const Scalar _tmp3 = 2 * _tmp2;
  const Scalar _tmp4 = _tmp1 - _tmp3;
  const Scalar _tmp5 = -_tmp4;
  const Scalar _tmp6 = _tmp5 * point(1, 0);
  const Scalar _tmp7 = _cam_T_world[1] * _tmp0;
  const Scalar _tmp8 = 2 * _cam_T_world[2];
  const Scalar _tmp9 = _cam_T_world[0] * _tmp8;
  const Scalar _tmp10 = _tmp7 + _tmp9;
  const Scalar _tmp11 = _tmp10 * point(2, 0);
  const Scalar _tmp12 = std::pow(_cam_T_world[2], Scalar(2));
  const Scalar _tmp13 = 2 * _tmp12;
  const Scalar _tmp14 = std::pow(_cam_T_world[1], Scalar(2));
  const Scalar _tmp15 = 2 * _tmp14 - 1;
  const Scalar _tmp16 = -_tmp13 - _tmp15;
  const Scalar _tmp17 = _tmp16 * point(0, 0);
  const Scalar _tmp18 = _cam_T_world[4] + _tmp11 + _tmp17 + _tmp6;
  const Scalar _tmp19 = _tmp7 - _tmp9;
  const Scalar _tmp20 = -_tmp19;
  const Scalar _tmp21 = _cam_T_world[0] * _tmp0;
  const Scalar _tmp22 = _cam_T_world[1] * _tmp8;
  const Scalar _tmp23 = _tmp21 + _tmp22;
  const Scalar _tmp24 = std::pow(_cam_T_world[0], Scalar(2));
  const Scalar _tmp25 = 2 * _tmp24;
  const Scalar _tmp26 = -_tmp15 - _tmp25;
  const Scalar _tmp27 =
      -_cam_T_world[6] - _tmp20 * point(0, 0) - _tmp23 * point(1, 0) - _tmp26 * point(2, 0);
  const Scalar _tmp28 = Scalar(1.0) / (_tmp27);
  const Scalar _tmp29 = std::pow(_tmp18, Scalar(2));
  const Scalar _tmp30 = std::pow(_tmp27, Scalar(-2));
  const Scalar _tmp31 = _tmp29 * _tmp30;
  const Scalar _tmp32 = _tmp1 + _tmp3;
  const Scalar _tmp33 = _tmp32 * point(0, 0);
  const Scalar _tmp34 = _tmp21 - _tmp22;
  const Scalar _tmp35 = -_tmp34;
  const Scalar _tmp36 = _tmp35 * point(2, 0);
  const Scalar _tmp37 = -_tmp13 - _tmp25 + 1;
  const Scalar _tmp38 = _tmp37 * point(1, 0);
  const Scalar _tmp39 = _cam_T_world[5] + _tmp33 + _tmp36 + _tmp38;
  const Scalar _tmp40 = std::pow(_tmp39, Scalar(2));
  const Scalar _tmp41 = _tmp30 * _tmp40;
  const Scalar _tmp42 = _tmp31 + _tmp41;
  const Scalar _tmp43 = std::pow(_tmp42, Scalar(2));
  const Scalar _tmp44 = _tmp42 * intrinsics(1, 0) + _tmp43 * intrinsics(2, 0) + 1;
  const Scalar _tmp45 = _tmp28 * _tmp44;
  const Scalar _tmp46 = _tmp45 * intrinsics(0, 0);
  const Scalar _tmp47 = _tmp18 * _tmp46 - pixel(0, 0);
  const Scalar _tmp48 = _tmp39 * _tmp46 - pixel(1, 0);
  const Scalar _tmp49 = _tmp10 * point(1, 0);
  const Scalar _tmp50 = _tmp4 * point(2, 0);
  const Scalar _tmp51 = -_tmp23 * point(2, 0);
  const Scalar _tmp52 = -_tmp14;
  const Scalar _tmp53 = std::pow(_cam_T_world[3], Scalar(2));
  const Scalar _tmp54 = -_tmp24 + _tmp53;
  const Scalar _tmp55 = _tmp12 + _tmp52 + _tmp54;
  const Scalar _tmp56 = _tmp55 * point(1, 0);
  const Scalar _tmp57 = _tmp51 + _tmp56;
  const Scalar _tmp58 = _tmp18 * _tmp30;
  const Scalar _tmp59 = _tmp44 * intrinsics(0, 0);
  const Scalar _tmp60 = _tmp58 * _tmp59;
  const Scalar _tmp61 = _tmp58 * (2 * _tmp49 + 2 * _tmp50);
  const Scalar _tmp62 = _tmp35 * point(1, 0);
  const Scalar _tmp63 = -_tmp12;
  const Scalar _tmp64 = _tmp14 + _tmp54 + _tmp63;
  const Scalar _tmp65 = -_tmp64 * point(2, 0);
  const Scalar _tmp66 = _tmp30 * _tmp39;
  const Scalar _tmp67 = _tmp66 * (2 * _tmp62 + 2 * _tmp65);
  const Scalar _tmp68 = std::pow(_tmp27, Scalar(-3));
  const Scalar _tmp69 = _tmp68 * (2 * _tmp51 + 2 * _tmp56);
  const Scalar _tmp70 = _tmp29 * _tmp69;
  const Scalar _tmp71 = _tmp40 * _tmp69;
  const Scalar _tmp72 = _tmp42 * intrinsics(2, 0);
  const Scalar _tmp73 = _tmp28 * intrinsics(0, 0);
  const Scalar _tmp74 = _tmp73 * (_tmp72 * (2 * _tmp61 + 2 * _tmp67 + 2 * _tmp70 + 2 * _tmp71) +
                                  intrinsics(1, 0) * (_tmp61 + _tmp67 + _tmp70 + _tmp71));
  const Scalar _tmp75 = _tmp18 * _tmp74 + _tmp46 * (_tmp49 + _tmp50) + _tmp57 * _tmp60;
  const Scalar _tmp76 = _tmp59 * _tmp66;
  const Scalar _tmp77 = _tmp39 * _tmp74 + _tmp46 * (_tmp62 + _tmp65) + _tmp57 * _tmp76;
  const Scalar _tmp78 = -_tmp10 * point(0, 0);
  const Scalar _tmp79 = _tmp24 + _tmp52 + _tmp53 + _tmp63;
  const Scalar _tmp80 = _tmp79 * point(2, 0);
  const Scalar _tmp81 = _tmp20 * point(2, 0);
  const Scalar _tmp82 = -_tmp55 * point(0, 0);
  const Scalar _tmp83 = _tmp81 + _tmp82;
  const Scalar _tmp84 = _tmp34 * point(0, 0);
  const Scalar _tmp85 = _tmp32 * point(2, 0);
  const Scalar _tmp86 = _tmp66 * (2 * _tmp84 + 2 * _tmp85);
  const Scalar _tmp87 = _tmp58 * (2 * _tmp78 + 2 * _tmp80);
  const Scalar _tmp88 = _tmp68 * (2 * _tmp81 + 2 * _tmp82);
  const Scalar _tmp89 = _tmp29 * _tmp88;
  const Scalar _tmp90 = _tmp40 * _tmp88;
  const Scalar _tmp91 = _tmp73 * (_tmp72 * (2 * _tmp86 + 2 * _tmp87 + 2 * _tmp89 + 2 * _tmp90) +
                                  intrinsics(1, 0) * (_tmp86 + _tmp87 + _tmp89 + _tmp90));
  const Scalar _tmp92 = _tmp18 * _tmp91 + _tmp46 * (_tmp78 + _tmp80) + _tmp60 * _tmp83;
  const Scalar _tmp93 = _tmp39 * _tmp91 + _tmp46 * (_tmp84 + _tmp85) + _tmp76 * _tmp83;
  const Scalar _tmp94 = _tmp5 * point(0, 0);
  const Scalar _tmp95 = -_tmp79 * point(1, 0);
  const Scalar _tmp96 = _tmp23 * point(0, 0);
  const Scalar _tmp97 = _tmp19 * point(1, 0);
  const Scalar _tmp98 = _tmp96 + _tmp97;
  const Scalar _tmp99 = _tmp68 * (2 * _tmp96 + 2 * _tmp97);
  const Scalar _tmp100 = _tmp29 * _tmp99;
  const Scalar _tmp101 = _tmp40 * _tmp99;
  const Scalar _tmp102 = _tmp58 * (2 * _tmp94 + 2 * _tmp95);
  const Scalar _tmp103 = -_tmp32 * point(1, 0);
  const Scalar _tmp104 = _tmp64 * point(0, 0);
  const Scalar _tmp105 = _tmp66 * (2 * _tmp103 + 2 * _tmp104);
  const Scalar _tmp106 =
      _tmp73 * (_tmp72 * (2 * _tmp100 + 2 * _tmp101 + 2 * _tmp102 + 2 * _tmp105) +
                intrinsics(1, 0) * (_tmp100 + _tmp101 + _tmp102 + _tmp105));
  const Scalar _tmp107 = _tmp106 * _tmp18 + _tmp46 * (_tmp94 + _tmp95) + _tmp60 * _tmp98;
  const Scalar _tmp108 = _tmp106 * _tmp39 + _tmp46 * (_tmp103 + _tmp104) + _tmp76 * _tmp98;
  const Scalar _tmp109 = 2 * _cam_T_world[4] + 2 * _tmp11 + 2 * _tmp17 + 2 * _tmp6;
  const Scalar _tmp110 = _tmp30 * intrinsics(1, 0);
  const Scalar _tmp111 = 2 * _tmp30 * _tmp72;
  const Scalar _tmp112 = _tmp109 * _tmp110 + _tmp109 * _tmp111;
  const Scalar _tmp113 = _tmp112 * _tmp73;
  const Scalar _tmp114 = _tmp113 * _tmp18 + _tmp46;
  const Scalar _tmp115 = _tmp113 * _tmp39;
  const Scalar _tmp116 = 2 * _cam_T_world[5] + 2 * _tmp33 + 2 * _tmp36 + 2 * _tmp38;
  const Scalar _tmp117 = _tmp110 * _tmp116 + _tmp111 * _tmp116;
  const Scalar _tmp118 = _tmp117 * _tmp73;
  const Scalar _tmp119 = _tmp118 * _tmp18;
  const Scalar _tmp120 = _tmp118 * _tmp39 + _tmp46;
  const Scalar _tmp121 = 2 * _tmp68;
  const Scalar _tmp122 = _tmp121 * _tmp29;
  const Scalar _tmp123 = _tmp121 * _tmp40;
  const Scalar _tmp124 = 4 * _tmp68;
  const Scalar _tmp125 = _tmp73 * (_tmp72 * (_tmp124 * _tmp29 + _tmp124 * _tmp40) +
                                   intrinsics(1, 0) * (_tmp122 + _tmp123));
  const Scalar _tmp126 = _tmp125 * _tmp18 + _tmp60;
  const Scalar _tmp127 = _tmp125 * _tmp39 + _tmp76;
  const Scalar _tmp128 = _tmp18 * _tmp45;
  const Scalar _tmp129 = _tmp39 * _tmp45;
  const Scalar _tmp130 = _tmp42 * _tmp73;
  const Scalar _tmp131 = _tmp130 * _tmp18;
  const Scalar _tmp132 = _tmp130 * _tmp39;
  const Scalar _tmp133 = _tmp43 * _tmp73;
  const Scalar _tmp134 = _tmp133 * _tmp18;
  const Scalar _tmp135 = _tmp133 * _tmp39;
  const Scalar _tmp136 = 4 * _cam_T_world[3];
  const Scalar _tmp137 = _cam_T_world[2] * _tmp136;
  const Scalar _tmp138 = _tmp66 * (_tmp137 + 4 * _tmp2);
  const Scalar _tmp139 = _cam_T_world[1] * _tmp136;
  const Scalar _tmp140 = 4 * _cam_T_world[0] * _cam_T_world[2] - _tmp139;
  const Scalar _tmp141 = _tmp140 * _tmp68;
  const Scalar _tmp142 = 4 * _tmp14;
  const Scalar _tmp143 = 4 * _tmp12 - 2;
  const Scalar _tmp144 = _tmp58 * (-_tmp142 - _tmp143);
  const Scalar _tmp145 =
      _tmp73 * (_tmp72 * (_tmp122 * _tmp140 + _tmp123 * _tmp140 + 2 * _tmp138 + 2 * _tmp144) +
                intrinsics(1, 0) * (_tmp138 + _tmp141 * _tmp29 + _tmp141 * _tmp40 + _tmp144));
  const Scalar _tmp146 = _tmp145 * _tmp18 + _tmp16 * _tmp46 + _tmp20 * _tmp60;
  const Scalar _tmp147 = _tmp145 * _tmp39 + _tmp20 * _tmp76 + _tmp32 * _tmp46;
  const Scalar _tmp148 = _tmp58 * (4 * _cam_T_world[0] * _cam_T_world[1] - _tmp137);
  const Scalar _tmp149 = _cam_T_world[0] * _tmp136;
  const Scalar _tmp150 = 4 * _cam_T_world[2];
  const Scalar _tmp151 = _cam_T_world[1] * _tmp150 + _tmp149;
  const Scalar _tmp152 = _tmp151 * _tmp68;
  const Scalar _tmp153 = 4 * _tmp24;
  const Scalar _tmp154 = _tmp66 * (-_tmp143 - _tmp153);
  const Scalar _tmp155 =
      _tmp73 * (_tmp72 * (_tmp122 * _tmp151 + _tmp123 * _tmp151 + 2 * _tmp148 + 2 * _tmp154) +
                intrinsics(1, 0) * (_tmp148 + _tmp152 * _tmp29 + _tmp152 * _tmp40 + _tmp154));
  const Scalar _tmp156 = _tmp155 * _tmp18 + _tmp23 * _tmp60 + _tmp46 * _tmp5;
  const Scalar _tmp157 = _tmp155 * _tmp39 + _tmp23 * _tmp76 + _tmp37 * _tmp46;
  const Scalar _tmp158 = _tmp66 * (4 * _cam_T_world[1] * _cam_T_world[2] - _tmp149);
  const Scalar _tmp159 = _tmp58 * (_cam_T_world[0] * _tmp150 + _tmp139);
  const Scalar _tmp160 = -_tmp142 - _tmp153 + 2;
  const Scalar _tmp161 = _tmp160 * _tmp68;
  const Scalar _tmp162 =
      _tmp73 * (_tmp72 * (_tmp122 * _tmp160 + _tmp123 * _tmp160 + 2 * _tmp158 + 2 * _tmp159) +
                intrinsics(1, 0) * (_tmp158 + _tmp159 + _tmp161 * _tmp29 + _tmp161 * _tmp40));
  const Scalar _tmp163 = _tmp10 * _tmp46 + _tmp162 * _tmp18 + _tmp26 * _tmp60;
  const Scalar _tmp164 = _tmp162 * _tmp39 + _tmp26 * _tmp76 + _tmp35 * _tmp46;
  const Scalar _tmp165 = std::pow(intrinsics(0, 0), Scalar(2));
  const Scalar _tmp166 = _tmp165 * _tmp41;
  const Scalar _tmp167 = _tmp41 * _tmp59;
  const Scalar _tmp168 = _tmp166 * _tmp43;
  const Scalar _tmp169 = _tmp165 * _tmp31;
  const Scalar _tmp170 = _tmp31 * _tmp59;
  const Scalar _tmp171 = _tmp169 * _tmp43;
  const Scalar _tmp172 = std::pow(_tmp44, Scalar(2));
  const Scalar _tmp173 = [&]() {
    const Scalar base = _tmp42;
    return base * base * base;
  }();
  const Scalar _tmp174 = std::pow(_tmp42, Scalar(4));

  // Output terms (4)
  if (res != nullptr) {
    Eigen::Matrix<Scalar, 2, 1>& _res = (*res);

    _res(0, 0) = _tmp47;
    _res(1, 0) = _tmp48;
  }

  if (jacobian != nullptr) {
    Eigen::Matrix<Scalar, 2, 12>& _jacobian = (*jacobian);

    _jacobian(0, 0) = _tmp75;
    _jacobian(1, 0) = _tmp77;
    _jacobian(0, 1) = _tmp92;
    _jacobian(1, 1) = _tmp93;
    _jacobian(0, 2) = _tmp107;
    _jacobian(1, 2) = _tmp108;
    _jacobian(0, 3) = _tmp114;
    _jacobian(1, 3) = _tmp115;
    _jacobian(0, 4) = _tmp119;
    _jacobian(1, 4) = _tmp120;
    _jacobian(0, 5) = _tmp126;
    _jacobian(1, 5) = _tmp127;
    _jacobian(0, 6) = _tmp128;
    _jacobian(1, 6) = _tmp129;
    _jacobian(0, 7) = _tmp131;
    _jacobian(1, 7) = _tmp132;
    _jacobian(0, 8) = _tmp134;
    _jacobian(1, 8) = _tmp135;
    _jacobian(0, 9) = _tmp146;
    _jacobian(1, 9) = _tmp147;
    _jacobian(0, 10) = _tmp156;
    _jacobian(1, 10) = _tmp157;
    _jacobian(0, 11) = _tmp163;
    _jacobian(1, 11) = _tmp164;
  }

  if (hessian != nullptr) {
    Eigen::Matrix<Scalar, 12, 12>& _hessian = (*hessian);

    _hessian(0, 0) = std::pow(_tmp75, Scalar(2)) + std::pow(_tmp77, Scalar(2));
    _hessian(1, 0) = _tmp75 * _tmp92 + _tmp77 * _tmp93;
    _hessian(2, 0) = _tmp107 * _tmp75 + _tmp108 * _tmp77;
    _hessian(3, 0) = _tmp114 * _tmp75 + _tmp115 * _tmp77;
    _hessian(4, 0) = _tmp119 * _tmp75 + _tmp120 * _tmp77;
    _hessian(5, 0) = _tmp126 * _tmp75 + _tmp127 * _tmp77;
    _hessian(6, 0) = _tmp128 * _tmp75 + _tmp129 * _tmp77;
    _hessian(7, 0) = _tmp131 * _tmp75 + _tmp132 * _tmp77;
    _hessian(8, 0) = _tmp134 * _tmp75 + _tmp135 * _tmp77;
    _hessian(9, 0) = _tmp146 * _tmp75 + _tmp147 * _tmp77;
    _hessian(10, 0) = _tmp156 * _tmp75 + _tmp157 * _tmp77;
    _hessian(11, 0) = _tmp163 * _tmp75 + _tmp164 * _tmp77;
    _hessian(0, 1) = 0;
    _hessian(1, 1) = std::pow(_tmp92, Scalar(2)) + std::pow(_tmp93, Scalar(2));
    _hessian(2, 1) = _tmp107 * _tmp92 + _tmp108 * _tmp93;
    _hessian(3, 1) = _tmp114 * _tmp92 + _tmp115 * _tmp93;
    _hessian(4, 1) = _tmp119 * _tmp92 + _tmp120 * _tmp93;
    _hessian(5, 1) = _tmp126 * _tmp92 + _tmp127 * _tmp93;
    _hessian(6, 1) = _tmp128 * _tmp92 + _tmp129 * _tmp93;
    _hessian(7, 1) = _tmp131 * _tmp92 + _tmp132 * _tmp93;
    _hessian(8, 1) = _tmp134 * _tmp92 + _tmp135 * _tmp93;
    _hessian(9, 1) = _tmp146 * _tmp92 + _tmp147 * _tmp93;
    _hessian(10, 1) = _tmp156 * _tmp92 + _tmp157 * _tmp93;
    _hessian(11, 1) = _tmp163 * _tmp92 + _tmp164 * _tmp93;
    _hessian(0, 2) = 0;
    _hessian(1, 2) = 0;
    _hessian(2, 2) = std::pow(_tmp107, Scalar(2)) + std::pow(_tmp108, Scalar(2));
    _hessian(3, 2) = _tmp107 * _tmp114 + _tmp108 * _tmp115;
    _hessian(4, 2) = _tmp107 * _tmp119 + _tmp108 * _tmp120;
    _hessian(5, 2) = _tmp107 * _tmp126 + _tmp108 * _tmp127;
    _hessian(6, 2) = _tmp107 * _tmp128 + _tmp108 * _tmp129;
    _hessian(7, 2) = _tmp107 * _tmp131 + _tmp108 * _tmp132;
    _hessian(8, 2) = _tmp107 * _tmp134 + _tmp108 * _tmp135;
    _hessian(9, 2) = _tmp107 * _tmp146 + _tmp108 * _tmp147;
    _hessian(10, 2) = _tmp107 * _tmp156 + _tmp108 * _tmp157;
    _hessian(11, 2) = _tmp107 * _tmp163 + _tmp108 * _tmp164;
    _hessian(0, 3) = 0;
    _hessian(1, 3) = 0;
    _hessian(2, 3) = 0;
    _hessian(3, 3) = std::pow(_tmp112, Scalar(2)) * _tmp166 + std::pow(_tmp114, Scalar(2));
    _hessian(4, 3) = _tmp114 * _tmp119 + _tmp115 * _tmp120;
    _hessian(5, 3) = _tmp114 * _tmp126 + _tmp115 * _tmp127;
    _hessian(6, 3) = _tmp112 * _tmp167 + _tmp114 * _tmp128;
    _hessian(7, 3) = _tmp112 * _tmp166 * _tmp42 + _tmp114 * _tmp131;
    _hessian(8, 3) = _tmp112 * _tmp168 + _tmp114 * _tmp134;
    _hessian(9, 3) = _tmp114 * _tmp146 + _tmp115 * _tmp147;
    _hessian(10, 3) = _tmp114 * _tmp156 + _tmp115 * _tmp157;
    _hessian(11, 3) = _tmp114 * _tmp163 + _tmp115 * _tmp164;
    _hessian(0, 4) = 0;
    _hessian(1, 4) = 0;
    _hessian(2, 4) = 0;
    _hessian(3, 4) = 0;
    _hessian(4, 4) = std::pow(_tmp117, Scalar(2)) * _tmp169 + std::pow(_tmp120, Scalar(2));
    _hessian(5, 4) = _tmp119 * _tmp126 + _tmp120 * _tmp127;
    _hessian(6, 4) = _tmp117 * _tmp170 + _tmp120 * _tmp129;
    _hessian(7, 4) = _tmp117 * _tmp169 * _tmp42 + _tmp120 * _tmp132;
    _hessian(8, 4) = _tmp117 * _tmp171 + _tmp120 * _tmp135;
    _hessian(9, 4) = _tmp119 * _tmp146 + _tmp120 * _tmp147;
    _hessian(10, 4) = _tmp119 * _tmp156 + _tmp120 * _tmp157;
    _hessian(11, 4) = _tmp119 * _tmp163 + _tmp120 * _tmp164;
    _hessian(0, 5) = 0;
    _hessian(1, 5) = 0;
    _hessian(2, 5) = 0;
    _hessian(3, 5) = 0;
    _hessian(4, 5) = 0;
    _hessian(5, 5) = std::pow(_tmp126, Scalar(2)) + std::pow(_tmp127, Scalar(2));
    _hessian(6, 5) = _tmp126 * _tmp128 + _tmp127 * _tmp129;
    _hessian(7, 5) = _tmp126 * _tmp131 + _tmp127 * _tmp132;
    _hessian(8, 5) = _tmp126 * _tmp134 + _tmp127 * _tmp135;
    _hessian(9, 5) = _tmp126 * _tmp146 + _tmp127 * _tmp147;
    _hessian(10, 5) = _tmp126 * _tmp156 + _tmp127 * _tmp157;
    _hessian(11, 5) = _tmp126 * _tmp163 + _tmp127 * _tmp164;
    _hessian(0, 6) = 0;
    _hessian(1, 6) = 0;
    _hessian(2, 6) = 0;
    _hessian(3, 6) = 0;
    _hessian(4, 6) = 0;
    _hessian(5, 6) = 0;
    _hessian(6, 6) = _tmp172 * _tmp31 + _tmp172 * _tmp41;
    _hessian(7, 6) = _tmp167 * _tmp42 + _tmp170 * _tmp42;
    _hessian(8, 6) = _tmp167 * _tmp43 + _tmp170 * _tmp43;
    _hessian(9, 6) = _tmp128 * _tmp146 + _tmp129 * _tmp147;
    _hessian(10, 6) = _tmp128 * _tmp156 + _tmp129 * _tmp157;
    _hessian(11, 6) = _tmp128 * _tmp163 + _tmp129 * _tmp164;
    _hessian(0, 7) = 0;
    _hessian(1, 7) = 0;
    _hessian(2, 7) = 0;
    _hessian(3, 7) = 0;
    _hessian(4, 7) = 0;
    _hessian(5, 7) = 0;
    _hessian(6, 7) = 0;
    _hessian(7, 7) = _tmp168 + _tmp171;
    _hessian(8, 7) = _tmp166 * _tmp173 + _tmp169 * _tmp173;
    _hessian(9, 7) = _tmp131 * _tmp146 + _tmp132 * _tmp147;
    _hessian(10, 7) = _tmp131 * _tmp156 + _tmp132 * _tmp157;
    _hessian(11, 7) = _tmp131 * _tmp163 + _tmp132 * _tmp164;
    _hessian(0, 8) = 0;
    _hessian(1, 8) = 0;
    _hessian(2, 8) = 0;
    _hessian(3, 8) = 0;
    _hessian(4, 8) = 0;
    _hessian(5, 8) = 0;
    _hessian(6, 8) = 0;
    _hessian(7, 8) = 0;
    _hessian(8, 8) = _tmp166 * _tmp174 + _tmp169 * _tmp174;
    _hessian(9, 8) = _tmp134 * _tmp146 + _tmp135 * _tmp147;
    _hessian(10, 8) = _tmp134 * _tmp156 + _tmp135 * _tmp157;
    _hessian(11, 8) = _tmp134 * _tmp163 + _tmp135 * _tmp164;
    _hessian(0, 9) = 0;
    _hessian(1, 9) = 0;
    _hessian(2, 9) = 0;
    _hessian(3, 9) = 0;
    _hessian(4, 9) = 0;
    _hessian(5, 9) = 0;
    _hessian(6, 9) = 0;
    _hessian(7, 9) = 0;
    _hessian(8, 9) = 0;
    _hessian(9, 9) = std::pow(_tmp146, Scalar(2)) + std::pow(_tmp147, Scalar(2));
    _hessian(10, 9) = _tmp146 * _tmp156 + _tmp147 * _tmp157;
    _hessian(11, 9) = _tmp146 * _tmp163 + _tmp147 * _tmp164;
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
    _hessian(10, 10) = std::pow(_tmp156, Scalar(2)) + std::pow(_tmp157, Scalar(2));
    _hessian(11, 10) = _tmp156 * _tmp163 + _tmp157 * _tmp164;
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
    _hessian(11, 11) = std::pow(_tmp163, Scalar(2)) + std::pow(_tmp164, Scalar(2));
  }

  if (rhs != nullptr) {
    Eigen::Matrix<Scalar, 12, 1>& _rhs = (*rhs);

    _rhs(0, 0) = _tmp47 * _tmp75 + _tmp48 * _tmp77;
    _rhs(1, 0) = _tmp47 * _tmp92 + _tmp48 * _tmp93;
    _rhs(2, 0) = _tmp107 * _tmp47 + _tmp108 * _tmp48;
    _rhs(3, 0) = _tmp114 * _tmp47 + _tmp115 * _tmp48;
    _rhs(4, 0) = _tmp119 * _tmp47 + _tmp120 * _tmp48;
    _rhs(5, 0) = _tmp126 * _tmp47 + _tmp127 * _tmp48;
    _rhs(6, 0) = _tmp128 * _tmp47 + _tmp129 * _tmp48;
    _rhs(7, 0) = _tmp131 * _tmp47 + _tmp132 * _tmp48;
    _rhs(8, 0) = _tmp134 * _tmp47 + _tmp135 * _tmp48;
    _rhs(9, 0) = _tmp146 * _tmp47 + _tmp147 * _tmp48;
    _rhs(10, 0) = _tmp156 * _tmp47 + _tmp157 * _tmp48;
    _rhs(11, 0) = _tmp163 * _tmp47 + _tmp164 * _tmp48;
  }
}  // NOLINT(readability/fn_size)

// NOLINTNEXTLINE(readability/fn_size)
}  // namespace sym
