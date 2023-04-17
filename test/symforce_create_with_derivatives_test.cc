/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <iostream>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <symforce/opt/util.h>

#include "symforce_function_codegen_test_data/symengine/with_jacobians/compose_pose3_with_jacobian0.h"

TEMPLATE_TEST_CASE("Test compose numerical derivative", "[with_jacobians]", double, float) {
  using Scalar = TestType;

  constexpr const Scalar epsilon = 1e-7f;

  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  std::mt19937 gen(24365);
  for (size_t i = 0; i < 10000; i++) {
    const sym::Pose3<Scalar> a =
        sym::Pose3<Scalar>(sym::Rot3<Scalar>::Random(gen), Vector3::Random());
    const sym::Pose3<Scalar> b =
        sym::Pose3<Scalar>(sym::Rot3<Scalar>::Random(gen), Vector3::Random());
    const Eigen::Matrix<Scalar, 6, 6> numerical_jacobian = sym::NumericalDerivative(
        std::bind(&sym::GroupOps<sym::Pose3<Scalar>>::Compose, std::placeholders::_1, b), a,
        epsilon, std::sqrt(epsilon));

    Eigen::Matrix<Scalar, 6, 6> symforce_jacobian;
    const sym::Pose3<Scalar> symforce_result =
        sym::ComposePose3WithJacobian0(a, b, &symforce_jacobian);
    (void)symforce_result;

    CHECK(numerical_jacobian.isApprox(symforce_jacobian, 10 * std::sqrt(epsilon)));
  }
}
