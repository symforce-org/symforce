/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./sym_type_casters.h"

namespace pybind11 {
namespace detail {

constexpr descr<4> handle_sym_type_name<sym::Rot2d>::name;
constexpr descr<4> handle_sym_type_name<sym::Rot3d>::name;
constexpr descr<5> handle_sym_type_name<sym::Pose2d>::name;
constexpr descr<5> handle_sym_type_name<sym::Pose3d>::name;
constexpr descr<5> handle_sym_type_name<sym::Unit3d>::name;
constexpr descr<13> handle_sym_type_name<sym::ATANCameraCald>::name;
constexpr descr<21> handle_sym_type_name<sym::DoubleSphereCameraCald>::name;
constexpr descr<24> handle_sym_type_name<sym::EquirectangularCameraCald>::name;
constexpr descr<15> handle_sym_type_name<sym::LinearCameraCald>::name;
constexpr descr<19> handle_sym_type_name<sym::PolynomialCameraCald>::name;
constexpr descr<18> handle_sym_type_name<sym::SphericalCameraCald>::name;

}  // namespace detail
}  // namespace pybind11
