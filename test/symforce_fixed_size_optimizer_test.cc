/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <symforce/opt/fixed_size_optimizer.h>
#include <symforce/opt/levenberg_marquardt_solver.h>
#include <symforce/opt/linearization.h>
#include <symforce/opt/optimization_stats.h>
#include <symforce/opt/optimizer.h>
#include <symforce/opt/sparse_cholesky/sparse_cholesky_solver.h>

sym::optimizer_params_t DefaultLmParams() {
  auto params = sym::DefaultOptimizerParams();
  params.verbose = true;
  return params;
}

// Let x and y be optimized variables
struct fixed_size_opt_vars_t {
  double x;
  double y;
};

// Define how x and y are updated each iteration
class FixedSizeState
    : public sym::internal::LevenbergMarquardtStateBase<FixedSizeState, fixed_size_opt_vars_t,
                                                        Eigen::SparseMatrix<double>> {
 public:
  void UpdateNewFromInitImpl(const sym::VectorX<Scalar>& update, const Scalar epsilon) {
    const auto& init_values = this->Init().values;
    auto& new_values = this->New().values;
    new_values.x = init_values.x + update[0];
    new_values.y = init_values.y + update[1];
  }

  sym::values_t GetLcmTypeImpl(const ValuesType& values) const {
    return {sym::index_t{}, {values.x, values.y}};
  }
};

// Let a and b be constants in the linearization function
struct fixed_size_constants_t {
  double a;
  double b;
};

// The linearization is a function of the states and constants
void LinearizeFixedSizeValues(const fixed_size_opt_vars_t& states,
                              const fixed_size_constants_t& constants,
                              sym::SparseLinearizationd& linearization) {
  linearization.residual = Eigen::Vector2d{states.x - constants.a, states.y - constants.b};
  linearization.jacobian = Eigen::SparseMatrix<double>(2, 2);
  linearization.jacobian.setIdentity();
  linearization.hessian_lower = Eigen::SparseMatrix<double>(2, 2);
  linearization.hessian_lower.setIdentity();
  linearization.rhs = linearization.jacobian.transpose() * linearization.residual;
}

TEST_CASE("Check that we can change linearization type", "[fixed_size_optimizer]") {
  // Helpful aliases
  using LinearSolverType = sym::SparseCholeskySolver<Eigen::SparseMatrix<double>>;
  using NonlinearSolverType =
      sym::LevenbergMarquardtSolver<double, LinearSolverType, FixedSizeState>;
  using Optimizer = sym::FixedSizeOptimizer<double, NonlinearSolverType>;

  // Initial guess for optimized variables
  fixed_size_opt_vars_t states{1.0, 2.0};

  // Constants (non-optimized variables)
  fixed_size_constants_t constants{42.0, 43.0};

  // Capture references to the constants in a lambda function representing the linearize func.
  // By capturing the constants in this lambda, we don't have to save them in the LM state.
  // This pattern is useful when optimizing the same problem with different constants each time, as
  // this lambda can be recreated each time the constants change.
  Optimizer::LinearizeFunc linearize_func = [&constants](const fixed_size_opt_vars_t& states,
                                                         sym::SparseLinearizationd& linearization) {
    LinearizeFixedSizeValues(states, constants, linearization);
  };

  // Optimize
  Optimizer optimizer(DefaultLmParams(), "sym::Optimizer", sym::kDefaultEpsilond,
                      sym::SparseCholeskySolver<Eigen::SparseMatrix<double>>(
                          Eigen::NaturalOrdering<Eigen::SparseMatrix<double>::StorageIndex>()));

  Optimizer::Stats stats = optimizer.Optimize(states, linearize_func, -1, false);

  // Check results
  CHECK(stats.status == sym::optimization_status_t::SUCCESS);
  CHECK(stats.failure_reason == sym::levenberg_marquardt_solver_failure_reason_t::INVALID);
  CHECK(std::abs(states.x - constants.a) < 1e-3);
  CHECK(std::abs(states.y - constants.b) < 1e-3);
}
