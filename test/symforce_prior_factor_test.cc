/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include <sym/factors/prior_factor_rot3.h>
#include <symforce/opt/util.h>

TEST_CASE("Test zero residual for a noiseless prior factor", "[prior_factors]") {
  const double sigma = 1.0;
  const Eigen::Vector3d sigmas = Eigen::Vector3d::Constant(sigma);
  const Eigen::Matrix3d sqrt_info = sigmas.cwiseInverse().asDiagonal();
  const double epsilon = 1e-9;

  std::mt19937 gen(42);
  const sym::Rot3d a = sym::Rot3d::Random(gen);
  const sym::Rot3d b = a;

  Eigen::Matrix<double, 3, 1> residual;
  Eigen::Matrix<double, 3, 3> jacobian;
  sym::PriorFactorRot3<double>(a, b, sqrt_info, epsilon, &residual, &jacobian);

  CHECK(residual.isZero(epsilon));
}

TEST_CASE("Test jacobian for noisy prior factors", "[prior_factors]") {
  const double sigma = 1.0;
  const Eigen::Vector3d sigmas = Eigen::Vector3d::Constant(sigma);
  const Eigen::Matrix3d sqrt_info = sigmas.cwiseInverse().asDiagonal();
  const double epsilon = 1e-9;
  const double delta = std::sqrt(epsilon);

  std::mt19937 gen(42);
  for (int i = 0; i < 10000; i++) {
    const sym::Rot3d a = sym::Rot3d::Random(gen);
    const sym::Rot3d b = sym::Rot3d::Random(gen);

    const double angle_between = a.Between(b).AngleAxis().angle();
    if (std::abs(angle_between - M_PI) < 2 * delta) {
      // skip this one if the angle between is too close to 180 deg
      continue;
    }

    Eigen::Matrix<double, 3, 1> residual;
    Eigen::Matrix<double, 3, 3> jacobian;
    sym::PriorFactorRot3<double>(a, b, sqrt_info, epsilon, &residual, &jacobian);

    const auto wrapped_residual = [&b, &sqrt_info, epsilon](const sym::Rot3d& a) {
      Eigen::Matrix<double, 3, 1> residual;
      Eigen::Matrix<double, 3, 3> jacobian;
      sym::PriorFactorRot3<double>(a, b, sqrt_info, epsilon, &residual, &jacobian);
      return residual;
    };

    const Eigen::Matrix<double, 3, 3> numerical_jacobian =
        sym::NumericalDerivative(wrapped_residual, a, epsilon, delta);

    CHECK(numerical_jacobian.isApprox(jacobian, delta));
  }
}
