#include <iostream>

#include <sym/factors/prior_factor_rot3.h>
#include <symforce/opt/util.h>

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

/**
 * Test zero residual for a noiseless prior factor.
 */
void TestZeroResidualPriorFactor() {
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

  assertTrue(residual.isZero(epsilon));
}

/**
 * Test jacobian for noisy prior factors
 */
void TestPriorFactorJacobian() {
  const double sigma = 1.0;
  const Eigen::Vector3d sigmas = Eigen::Vector3d::Constant(sigma);
  const Eigen::Matrix3d sqrt_info = sigmas.cwiseInverse().asDiagonal();
  const double epsilon = 1e-9;

  std::mt19937 gen(42);
  for (int i = 0; i < 10000; i++) {
    const sym::Rot3d a = sym::Rot3d::Random(gen);
    const sym::Rot3d b = sym::Rot3d::Random(gen);

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
        sym::NumericalDerivative(wrapped_residual, a, epsilon, std::sqrt(epsilon));

    assertTrue(numerical_jacobian.isApprox(jacobian, std::sqrt(epsilon)));
  }
}

int main(int argc, char** argv) {
  TestZeroResidualPriorFactor();
  TestPriorFactorJacobian();
}
