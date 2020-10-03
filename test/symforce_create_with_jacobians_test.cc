#include <iostream>

#include <symforce/opt/util.h>

#include "symforce_function_codegen_test_data/create_with_jacobians/compose_pose3__value_and_jacobian0.h"

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

namespace sym {

template <class Scalar>
void TestComposeNumericalDerivative() {
  constexpr const Scalar epsilon = 1e-7f;

  using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
  std::mt19937 gen(24365);
  for (size_t i = 0; i < 10000; i++) {
    const geo::Pose3<Scalar> a =
        geo::Pose3<Scalar>(geo::Rot3<Scalar>::Random(gen), Vector3::Random());
    const geo::Pose3<Scalar> b =
        geo::Pose3<Scalar>(geo::Rot3<Scalar>::Random(gen), Vector3::Random());
    const Eigen::Matrix<Scalar, 6, 6> numerical_jacobian = sym::NumericalDerivative(
        std::bind(&geo::GroupOps<geo::Pose3<Scalar>>::Compose, std::placeholders::_1, b), a,
        epsilon, std::sqrt(epsilon));

    Eigen::Matrix<Scalar, 6, 6> symforce_jacobian;
    const geo::Pose3<Scalar> symforce_result =
        sym::ComposePose3_ValueAndJacobian0(a, b, &symforce_jacobian);
    (void)symforce_result;

    assertTrue(numerical_jacobian.isApprox(symforce_jacobian, 10 * std::sqrt(epsilon)));
  }
}

}  // namespace sym

int main(int argc, char** argv) {
  sym::TestComposeNumericalDerivative<double>();
  sym::TestComposeNumericalDerivative<float>();
}
