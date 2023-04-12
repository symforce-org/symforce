/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <random>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

#include <sym/ops/storage_ops.h>
#include <symforce/opt/dense_linearizer.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/key.h>
#include <symforce/opt/linearization.h>

TEST_CASE("Correct hessian when factor key order doesn't match problem order",
          "[dense-linearizer]") {
  using M3 = Eigen::Matrix3d;

  // J: 2 0 0  JtJ: 5 0 1
  //    0 3 1       0 9 3
  //    1 0 1       1 3 2
  // Chosen so that if test fails, it'll be easier to visually inspect the hessian and JtJ to see
  // what went wrong.
  const M3 J = (M3() << 2, 0, 0, 0, 3, 1, 1, 0, 1).finished();
  const std::vector<sym::Key> keys = {'x', 'y', 'z'};
  const std::vector<sym::Factord> factors = {sym::Factord::Jacobian(
      [&](const double a, const double b, const double c, Eigen::Vector3d* const res,
          M3* const jac) {
        if (res != nullptr) {
          *res = J * Eigen::Vector3d(a, b, c);
        }
        if (jac != nullptr) {
          *jac = J;
        }
      },
      keys)};

  const sym::Valuesd values = [&] {
    sym::Valuesd out;
    out.Set<double>(keys[0], 7);
    out.Set<double>(keys[1], 11);
    out.Set<double>(keys[2], 13);
    return out;
  }();

  // Since we swapped the order of the first two keys, the jacobian and hessian should be
  // J: 0 2 0   JtJ:  9 0 3
  //    3 0 1         0 5 1
  //    0 1 1         3 1 2
  sym::DenseLinearizer<double> linearizer("linearizer", factors, {keys[1], keys[0], keys[2]},
                                          true /* include_jacobians */);
  sym::DenseLinearization<double> linearization;
  linearizer.Relinearize(values, linearization);

  const M3 expected_J = (M3() << 0, 2, 0, 3, 0, 1, 0, 1, 1).finished();
  const M3 expected_H = (M3() << 9, 0, 3, 0, 5, 1, 3, 1, 2).finished();

  CHECK(expected_J == linearization.jacobian);
  CHECK(expected_H == M3(linearization.hessian_lower.template selfadjointView<Eigen::Lower>()));
}

TEST_CASE("Residual, jacobian, hessian, and rhs are all consistent", "[dense-linearizer]") {
  // Abbreviate types for readability
  using M24 = Eigen::Matrix<double, 2, 4>;
  using V2 = Eigen::Vector2d;
  using V4 = Eigen::Vector4d;
  using V8 = Eigen::Matrix<double, 8, 1>;
  using M84 = Eigen::Matrix<double, 8, 4>;
  using M66 = Eigen::Matrix<double, 6, 6>;

  std::mt19937 gen(7919);

  // Set up problem
  std::vector<sym::Factord> factors;
  const M24 J24 = sym::StorageOps<M24>::Random(gen);
  factors.push_back(sym::Factord::Jacobian(
      [J24](const V2& a, const V2& b, V2* const res, M24* const jac) {
        if (res != nullptr) {
          *res = J24 * (V4() << a, b).finished();
        }
        if (jac != nullptr) {
          *jac = J24;
        }
      },
      {'y', 'x'}));
  const M84 J84 = sym::StorageOps<M84>::Random(gen);
  factors.push_back(sym::Factord::Jacobian(
      [J84](const V2& a, const V2& b, V8* const res, M84* const jac) {
        if (res != nullptr) {
          *res = J84 * (V4() << a, b).finished();
        }
        if (jac != nullptr) {
          *jac = J84;
        }
      },
      {'z', 'y'}));

  const sym::Valuesd values = [] {
    sym::Valuesd out;
    out.Set<V2>('x', V2(2, 3));
    out.Set<V2>('y', V2(5, 7));
    out.Set<V2>('z', V2(9, 11));
    return out;
  }();

  sym::DenseLinearizer<double> linearizer("linearizer", factors, {'x', 'y', 'z'},
                                          true /* include_jacobians */);
  sym::DenseLinearization<double> linearization;
  linearizer.Relinearize(values, linearization);

  const Eigen::Matrix<double, 10, 1> res = linearization.residual;
  const Eigen::Matrix<double, 10, 6> jac = linearization.jacobian;
  const auto hes = M66(linearization.hessian_lower.template selfadjointView<Eigen::Lower>());
  const Eigen::Matrix<double, 6, 1> rhs = linearization.rhs;

  // Check that linearization is self consistent
  CHECK((hes - jac.transpose() * jac).cwiseAbs().maxCoeff() < 1e-13);
  CHECK((rhs - jac.transpose() * res).cwiseAbs().maxCoeff() < 1e-13);

  // Check that the initial linearization matches subsequent linearizations
  linearizer.Relinearize(values, linearization);
  CHECK((res - linearization.residual).cwiseAbs().maxCoeff() < 1e-15);
  CHECK((jac - linearization.jacobian).cwiseAbs().maxCoeff() < 1e-15);
  CHECK((hes - M66(linearization.hessian_lower.template selfadjointView<Eigen::Lower>()))
            .cwiseAbs()
            .maxCoeff() < 1e-15);
  CHECK((rhs - linearization.rhs).cwiseAbs().maxCoeff() < 1e-15);
}

TEST_CASE("Jacobian is not allocated if include_jacobians is false", "[dense-linearizer]") {
  using M3 = Eigen::Matrix3d;
  using V3 = Eigen::Vector3d;

  // Set up a dummy problem. The details don't matter.
  const std::vector<sym::Factord> factors = {sym::Factord::Jacobian(
      [&](const V3 a, V3* const res, M3* const jac) {
        if (res != nullptr) {
          *res = a;
        }
        if (jac != nullptr) {
          *jac = M3::Identity();
        }
      },
      {'x'})};

  const sym::Valuesd values = [&] {
    sym::Valuesd out;
    out.Set<V3>('x', V3(2, 3, 5));
    return out;
  }();

  sym::DenseLinearizer<double> linearizer("lin", factors, {'x'}, false /* include_jacobians */);
  sym::DenseLinearization<double> linearization;
  linearizer.Relinearize(values, linearization);

  CHECK(linearization.jacobian.size() == 0);
}
