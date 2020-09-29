#include <iostream>

#include <sym/factors/between_factor_rot3.h>
#include <symforce/opt/util.h>

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

/**
 * Test zero residual for a noiseless between factor.
 */
void TestZeroResidualBetweenFactor() {
  const double sigma = 1.0;
  const Eigen::Vector3d sigmas = Eigen::Vector3d::Constant(sigma);
  const Eigen::Matrix3d sqrt_info = sigmas.cwiseInverse().asDiagonal();
  const double epsilon = 1e-9;

  std::mt19937 gen(42);
  const geo::Rot3d a = geo::Rot3d::Random(gen);
  const geo::Rot3d b = geo::Rot3d::Random(gen);
  const geo::Rot3d a_T_b = a.Between(b);

  Eigen::Matrix<double, 3, 1> residual;
  Eigen::Matrix<double, 3, 6> jacobian;
  sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, &residual, &jacobian);

  assertTrue(residual.isZero(epsilon));
}

/**
 * Test jacobian for noisy between factors
 */
void TestBetweenFactorJacobian() {
  const double sigma = 1.0;
  const Eigen::Vector3d sigmas = Eigen::Vector3d::Constant(sigma);
  const Eigen::Matrix3d sqrt_info = sigmas.cwiseInverse().asDiagonal();
  const double epsilon = 1e-9;

  std::mt19937 gen(42);
  for (int i = 0; i < 10000; i++) {
    const geo::Rot3d a = geo::Rot3d::Random(gen);
    const geo::Rot3d b = geo::Rot3d::Random(gen);
    const geo::Rot3d a_T_b = geo::Rot3d::Random(gen);

    Eigen::Matrix<double, 3, 1> residual;
    Eigen::Matrix<double, 3, 6> jacobian;
    sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, &residual, &jacobian);

    const auto wrapped_residual_a = [&b, &a_T_b, &sqrt_info, epsilon](const geo::Rot3d& a) {
      Eigen::Matrix<double, 3, 1> residual;
      Eigen::Matrix<double, 3, 6> jacobian;
      sym::BetweenFactorRot3<double>(a, b, a_T_b, sqrt_info, epsilon, &residual, &jacobian);
      return residual;
    };

    const auto wrapped_residual_b = [&a, &a_T_b, &sqrt_info, epsilon](const geo::Rot3d& b) {
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

    assertTrue(numerical_jacobian.isApprox(jacobian, std::sqrt(epsilon)));
  }
}

int main(int argc, char** argv) {
  TestZeroResidualBetweenFactor();
  TestBetweenFactorJacobian();
}
