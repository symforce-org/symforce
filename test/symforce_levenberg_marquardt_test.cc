/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <spdlog/spdlog.h>

#include <symforce/opt/levenberg_marquardt_solver.h>

/**
 * Test that Gauss Newton converges to the exact result for a linear residual where the solution is
 * zero
 *
 * We generate a random matrix J, and then set the residual to r = J * x
 */
TEMPLATE_TEST_CASE("Converges for a linear problem in one iteration", "[levenberg_marquardt]",
                   double, float) {
  using Scalar = TestType;

  constexpr const int M = 9;
  constexpr const int N = 5;

  std::mt19937 gen(12345);
  const sym::MatrixX<Scalar> J_MN = sym::Random<Eigen::Matrix<Scalar, M, N>>(gen);

  spdlog::debug("J_MN:\n{}", J_MN);
  spdlog::debug("J_MN^T * J_MN:\n{}", (J_MN.transpose() * J_MN).eval());

  constexpr const Scalar kEpsilon = 1e-10;

  sym::optimizer_params_t params{};
  params.initial_lambda = 1.0;
  params.lambda_up_factor = 3.0;
  params.lambda_down_factor = 1.0 / 3.0;
  params.lambda_lower_bound = 0.0;
  params.lambda_upper_bound = 0.0;
  params.iterations = 1;
  params.use_diagonal_damping = false;
  params.use_unit_damping = false;
  sym::LevenbergMarquardtSolver<Scalar> solver(params, "", kEpsilon);

  using StateVector = Eigen::Matrix<Scalar, N, 1>;

  auto residual_func = [&](const sym::Values<Scalar>& values,
                           sym::Linearization<Scalar>& linearization) {
    const auto state_vec = values.template At<StateVector>('v');
    linearization.residual = J_MN * state_vec;
    linearization.hessian_lower = (J_MN.transpose() * J_MN).sparseView();
    linearization.jacobian = J_MN.sparseView();
    linearization.rhs = J_MN.transpose() * linearization.residual;
  };

  sym::Values<Scalar> values_init{};
  values_init.Set('v', (StateVector::Ones() * 100).eval());
  sym::index_t index = values_init.CreateIndex({'v'});
  sym::Linearization<Scalar> linearization{};

  residual_func(values_init, linearization);
  const sym::VectorX<Scalar> residual_init = linearization.residual;
  const Scalar error_init = 0.5 * residual_init.squaredNorm();
  spdlog::debug("values_init: {}\n", values_init);
  spdlog::debug("residual_init: {}\n", residual_init.transpose());
  spdlog::debug("error_init: {}\n", error_init);

  solver.SetIndex(index);
  solver.Reset(values_init);

  // Collect debug stats so that we have the final residual
  const bool debug_stats = true;

  // Do a single gauss-newton iteration
  sym::OptimizationStats<Scalar> stats{};
  solver.Iterate(residual_func, stats, debug_stats);

  const sym::VectorX<Scalar> residual_final =
      stats.iterations.back().residual.template cast<Scalar>();
  const Scalar error_final = 0.5 * residual_final.squaredNorm();
  spdlog::debug("values_final: {}\n", solver.GetBestValues());
  spdlog::debug("residual_final: {}\n", residual_final.transpose());
  spdlog::debug("error_final: {}\n", error_final);

  // Check initial error was high and final is zero
  CHECK(error_init > 10000.);
  CHECK(error_final < 1e-8);

  // Check solution is zero
  CHECK(solver.GetBestValues().template At<StateVector>('v').norm() < 1e-4);
}
