#include <Eigen/Dense>
#include <symforce/opt/util.h>

#include "catch.hpp"

TEST_CASE("Test big vector derivatives", "[numerical_derivative]") {
  using BigMatrix = Eigen::Matrix<double, 42, 1>;
  const auto identity_function = [](const BigMatrix& m) -> BigMatrix { return m; };
  const Eigen::Matrix<double, 42, 42> jacobian =
      sym::NumericalDerivative(identity_function, BigMatrix::Zero().eval());

  CHECK(jacobian.isApprox(Eigen::Matrix<double, 42, 42>::Identity(), 1e-10));
}

TEST_CASE("Test dynamic vector derivatives", "[numerical_derivative]") {
  const auto identity_function = [](const Eigen::VectorXd& v) -> Eigen::VectorXd { return v; };
  const Eigen::MatrixXd jacobian =
      sym::NumericalDerivative(identity_function, Eigen::VectorXd::Zero(100).eval());

  CHECK(jacobian.isApprox(Eigen::MatrixXd::Identity(100, 100), 1e-10));
}
