/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include <sym/factors/between_factor_rot3.h>
#include <symforce/opt/util.h>

TEST_CASE("Test zero residual for a noiseless between factor", "[between_factors]") {
  const double sigma = 1.0;
  const Eigen::Vector3d sigmas = Eigen::Vector3d::Constant(sigma);
  const Eigen::Matrix3d sqrt_info = sigmas.cwiseInverse().asDiagonal();
  const double epsilon = 1e-9;

  std::mt19937 gen(42);
  const sym::Rot3d a = sym::Rot3d::Random(gen);
  const sym::Rot3d b = sym::Rot3d::Random(gen);
  const sym::Rot3d a_T_b = a.Between(b);

  Eigen::Matrix<double, 3, 1> residual;
  Eigen::Matrix<double, 3, 6> jacobian;
  sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, &residual, &jacobian);

  CHECK(residual.isZero(epsilon));
}

TEST_CASE("Test jacobian for noisy between factors", "[between_factors]") {
  const double sigma = 1.0;
  const Eigen::Vector3d sigmas = Eigen::Vector3d::Constant(sigma);
  const Eigen::Matrix3d sqrt_info = sigmas.cwiseInverse().asDiagonal();
  const double epsilon = 1e-9;

  std::mt19937 gen(42);
  for (int i = 0; i < 10000; i++) {
    const sym::Rot3d a = sym::Rot3d::Random(gen);
    const sym::Rot3d b = sym::Rot3d::Random(gen);
    const sym::Rot3d a_T_b = sym::Rot3d::Random(gen);

    Eigen::Matrix<double, 3, 1> residual;
    Eigen::Matrix<double, 3, 6> jacobian;
    sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, &residual, &jacobian);

    const auto wrapped_residual_a = [&b, &a_T_b, &sqrt_info, epsilon](const sym::Rot3d& a) {
      Eigen::Matrix<double, 3, 1> residual;
      Eigen::Matrix<double, 3, 6> jacobian;
      sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, &residual, &jacobian);
      return residual;
    };

    const auto wrapped_residual_b = [&a, &a_T_b, &sqrt_info, epsilon](const sym::Rot3d& b) {
      Eigen::Matrix<double, 3, 1> residual;
      Eigen::Matrix<double, 3, 6> jacobian;
      sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, &residual, &jacobian);
      return residual;
    };

    Eigen::Matrix<double, 3, 6> numerical_jacobian;
    numerical_jacobian.leftCols<3>() =
        sym::NumericalDerivative(wrapped_residual_a, a, epsilon, std::sqrt(epsilon));
    numerical_jacobian.rightCols<3>() =
        sym::NumericalDerivative(wrapped_residual_b, b, epsilon, std::sqrt(epsilon));

    CHECK(numerical_jacobian.isApprox(jacobian, std::sqrt(epsilon)));
  }
}
