/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

#include <sym/rot3.h>
#include <symforce/opt/util.h>

TEST_CASE("Test interpolation", "[opt_util][interpolate]") {
  std::mt19937 gen(42);

  const Eigen::Vector3d axis = sym::Rot3d::Random(gen) * Eigen::Vector3d::UnitX();
  const double angle = M_PI / 3;

  const sym::Rot3d a = sym::Rot3d::Random(gen);
  const sym::Rot3d b = sym::Rot3d::FromAngleAxis(angle, axis) * a;

  const double alpha = 0.67;

  const sym::Rot3d expected_result = sym::Rot3d::FromAngleAxis(angle * alpha, axis) * a;

  CHECK(sym::Interpolate(a, b, alpha).IsApprox(expected_result, 1e-10));
  CHECK(sym::Interpolator<sym::Rot3d>{}(a, b, alpha).IsApprox(expected_result, 1e-10));

  CHECK(sym::Interpolate(a, b, alpha, 1e-10).IsApprox(expected_result, 1e-10));
  CHECK(sym::Interpolator<sym::Rot3d>{1e-10}(a, b, alpha).IsApprox(expected_result, 1e-10));
}

TEST_CASE("Test big vector derivatives", "[opt_util][numerical_derivative]") {
  using BigMatrix = Eigen::Matrix<double, 42, 1>;
  const auto identity_function = [](const BigMatrix& m) -> BigMatrix { return m; };
  const Eigen::Matrix<double, 42, 42> jacobian =
      sym::NumericalDerivative(identity_function, BigMatrix::Zero().eval());

  CHECK(jacobian.isApprox(Eigen::Matrix<double, 42, 42>::Identity(), 1e-10));
}

TEST_CASE("Test dynamic vector derivatives", "[opt_util][numerical_derivative]") {
  const auto identity_function = [](const Eigen::VectorXd& v) -> Eigen::VectorXd { return v; };
  const Eigen::MatrixXd jacobian =
      sym::NumericalDerivative(identity_function, Eigen::VectorXd::Zero(100).eval());

  CHECK(jacobian.isApprox(Eigen::MatrixXd::Identity(100, 100), 1e-10));
}
