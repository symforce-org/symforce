/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <symforce/opt/dense_cholesky_solver.h>
#include <symforce/opt/levenberg_marquardt_solver.h>
#include <symforce/opt/optimizer.h>

TEMPLATE_TEST_CASE("Can construct a LevenbergMarquardtSolver w/ DenseCholeskySolver",
                   "[DenseCholesky]", float, double) {
  using Scalar = TestType;

  sym::LevenbergMarquardtSolver<Scalar, sym::DenseCholeskySolver<Scalar>> lm_solver_double(
      sym::DefaultOptimizerParams(), "test_lm_solver", 1e-7);
}

TEST_CASE("Test that templated methods work", "[DenseCholesky]") {
  sym::DenseCholeskySolver<double> solver;
  const int N = 3;

  const Eigen::MatrixXd A = []() -> Eigen::MatrixXd {
    Eigen::MatrixXd L = Eigen::MatrixXd::Random(N, N);
    L.diagonal() = L.diagonal().cwiseAbs().array() + 1;
    return L * L.transpose();
  }();

  solver.Factorize(A);

  // Check Solve
  for (int i = 0; i < A.cols(); i++) {
    const Eigen::VectorXd rhs = Eigen::MatrixXd::Identity(N, N).col(i);
    const Eigen::VectorXd sol = solver.Solve(rhs);
    CHECK((A * sol - rhs).cwiseAbs().maxCoeff() < 1e-15);
  }

  // Check SolveInPlace
  Eigen::MatrixXd A_inv = Eigen::MatrixXd::Identity(N, N);
  solver.SolveInPlace(A_inv);
  CHECK((A * A_inv - Eigen::MatrixXd::Identity(N, N)).cwiseAbs().maxCoeff() < 1e-15);
}
