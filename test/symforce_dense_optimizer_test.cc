/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

#include <sym/util/epsilon.h>
#include <symforce/opt/dense_cholesky_solver.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/levenberg_marquardt_solver.h>
#include <symforce/opt/optimizer.h>
#include <symforce/opt/values.h>

template <typename Scalar>
using DenseOptimizer =
    sym::Optimizer<Scalar, sym::LevenbergMarquardtSolver<Scalar, sym::DenseCholeskySolver<Scalar>>>;

TEST_CASE("Optimizer can be used with dense cholesky solver", "[dense-optimizer]") {
  std::vector<sym::Factord> factors;

  factors.push_back(sym::Factord::Jacobian(
      [](double x, Eigen::Matrix<double, 1, 1>* res, Eigen::Matrix<double, 1, 1>* jac) {
        if (res) {
          (*res)(0) = 2 - x * x;
        }
        if (jac) {
          (*jac)(0, 0) = -2 * x;
        }
      },
      {'x'}));

  DenseOptimizer<double> optimizer(sym::DefaultOptimizerParams(), factors, sym::kDefaultEpsilond,
                                   "optimizer_name", {'x'}, true, true, true);

  sym::Valuesd values;
  values.Set('x', 2.0);

  DenseOptimizer<double>::Stats stats = optimizer.Optimize(values, -1, true);

  CHECK((std::sqrt(2.0) - values.At<double>('x')) < 1e-14);
}
