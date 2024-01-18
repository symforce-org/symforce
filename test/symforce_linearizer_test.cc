/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_test_macros.hpp>

#include <symforce/opt/assert.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/key.h>
#include <symforce/opt/linearization.h>
#include <symforce/opt/linearizer.h>

sym::Factord GetDenseFactor(const Eigen::Matrix2d& J, const std::vector<sym::Key>& keys) {
  return sym::Factord::Jacobian(
      [J](const double x, const double y, Eigen::Vector2d* const res, Eigen::Matrix2d* const jac) {
        if (res != nullptr) {
          *res = J * Eigen::Vector2d(x, y);
        }
        if (jac != nullptr) {
          *jac = J;
        }
      },
      keys);
}

sym::Factord GetSparseFactor(const Eigen::Matrix2d& J, const std::vector<sym::Key>& keys) {
  return sym::Factord::Jacobian(
      [J](const double x, const double y, Eigen::VectorXd* const res,
          Eigen::SparseMatrix<double>* const jac) {
        if (res != nullptr) {
          *res = J * Eigen::Vector2d(x, y);
        }
        if (jac != nullptr) {
          *jac = J.sparseView();
        }
      },
      keys);
}

sym::Valuesd GetValues(const std::vector<sym::Key>& keys) {
  SYM_ASSERT(keys.size() == 2);
  sym::Valuesd out;
  out.Set<double>(keys[0], 10);
  out.Set<double>(keys[1], -20);
  return out;
}

TEST_CASE("Column order of Jacobian matches key_order", "[linearizer]") {
  // Tests that the order of the columns in the jacobian matches the order
  // specified by the key_order argument of Linearizer
  const Eigen::Matrix2d J1 = (Eigen::Matrix2d() << 1, 0, 0, 2).finished();
  const Eigen::Matrix2d J2 = (Eigen::Matrix2d() << 3, 0, 0, 4).finished();
  const std::vector<sym::Key> keys({'x', 'y'});
  const std::vector<sym::Factord> factors = {GetDenseFactor(J1, keys), GetSparseFactor(J2, keys)};
  const sym::Valuesd values = GetValues(keys);

  using M42 = Eigen::Matrix<double, 4, 2>;
  const M42 J1J2 = (M42() << J1, J2).finished();

  // Key order 1
  {
    sym::Linearizer<double> linearizer("key_order_1", factors, keys, true /* include_jacobians */);
    sym::SparseLinearizationd linearization;
    linearizer.Relinearize(values, linearization);
    CHECK(M42(linearization.jacobian) == J1J2);
  }

  using M41 = Eigen::Matrix<double, 4, 1>;

  // Key order 2
  {
    sym::Linearizer<double> linearizer("key_order_2", factors, {keys[1], keys[0]},
                                       true /* include_jacobians */);
    sym::SparseLinearizationd linearization;
    linearizer.Relinearize(values, linearization);
    CHECK(M41(linearization.jacobian.col(0)) == M41(J1J2.col(1)));
    CHECK(M41(linearization.jacobian.col(1)) == M41(J1J2.col(0)));
  }
}

TEST_CASE("Relinearization is consistent w/ respect to factor order", "[linearizer]") {
  // Really two tests:
  // - Tests that Jt * J = hes and Jt * res = rhs (in the past, these relations were
  //   broken when a sparse factor preceded a dense factor in factor list)
  // - Tests that uninitialized relinearization gives same result as
  //   initialized relinearization.
  const Eigen::Matrix2d J1 = (Eigen::Matrix2d() << 1, 0, 0, 2).finished();
  const Eigen::Matrix2d J2 = (Eigen::Matrix2d() << 3, 0, 0, 4).finished();

  const std::vector<sym::Key> keys({{'x'}, {'y'}});
  const sym::Valuesd values = GetValues(keys);
  const std::vector<sym::Factord> dense_sparse = {GetDenseFactor(J1, keys),
                                                  GetSparseFactor(J2, keys)};
  const std::vector<sym::Factord> sparse_dense = {dense_sparse[1], dense_sparse[0]};

  for (const auto* const factors : {&dense_sparse, &sparse_dense}) {
    sym::Linearizer<double> linearizer("order_test", *factors, keys, true /* include_jacobians */);
    sym::SparseLinearizationd linearization;
    SYM_ASSERT(!linearization.IsInitialized());
    linearizer.Relinearize(values, linearization);

    // Copy is intentional
    const Eigen::Vector4d res = linearization.residual;
    const Eigen::Matrix<double, 4, 2> jac = linearization.jacobian;
    const Eigen::Matrix2d hes =
        Eigen::Matrix2d(linearization.hessian_lower).selfadjointView<Eigen::Lower>();
    const Eigen::Vector2d rhs = linearization.rhs;

    // Check that the residual, jacobian, hessian, and rhs are consistent
    CHECK(jac.transpose() * jac == hes);
    CHECK(jac.transpose() * res == rhs);

    // Linearization is calculated differently once initialized
    SYM_ASSERT(linearization.IsInitialized());
    linearizer.Relinearize(values, linearization);

    // Check that subsequent relinearizations give the same value
    CHECK(linearization.residual == res);
    CHECK(Eigen::Matrix<double, 4, 2>(linearization.jacobian) == jac);
    CHECK(Eigen::Matrix2d(
              Eigen::Matrix2d(linearization.hessian_lower).selfadjointView<Eigen::Lower>()) == hes);
    CHECK(linearization.rhs == rhs);
  }
}
