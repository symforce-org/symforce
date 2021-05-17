#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ac_sparse_math/sparse_cholesky_solver.h>
#include <symforce/opt/values.h>

#include <lcmtypes/sym/optimization_stats_t.hpp>
#include <lcmtypes/sym/optimizer_params_t.hpp>

#include "./internal/levenberg_marquardt_state.h"

namespace sym {

/**
 * Fast Levenberg-Marquardt solver for nonlinear least squares problems specified by a
 * linearization function.  Supports on-manifold optimization and sparse solving, and attempts to
 * minimize allocations after the first iteration.
 *
 * This assumes the problem structure is the same for the lifetime of the object - if the problem
 * structure changes, create a new LevenbergMarquardtSolver.
 *
 * TODO(aaron): Analyze what repeated allocations we do have, and get rid of them if possible
 *
 * Not thread safe! Create one per thread.
 *
 * Example usage:
 *
 *   constexpr const int M = 9;
 *   constexpr const int N = 5;
 *
 *   // Create a function that computes the residual (a linear residual for this example)
 *   const auto J_MN = sym::RandomNormalMatrix<double, M, N>(gen);
 *   const auto linearize_func = [&J_MN](const sym::Valuesd& values,
 *                                       sym::Linearizationd* const linearization) {
 *     const auto state_vec = values.At<sym::Vector5d>('v');
 *     linearization->residual = J_MN * state_vec;
 *     linearization->hessian_lower = (J_MN.transpose() * J_MN).sparseView();
 *     linearization->jacobian = J_MN.sparseView();
 *     linearization->rhs = J_MN.transpose() * linearization->residual;
 *   };
 *
 *   // Create a Values
 *   sym::Valuesd values_init{};
 *   values_init.Set('v', (StateVector::Ones() * 100).eval());
 *
 *   // Create a Solver
 *   sym::LevenbergMarquardtSolverd solver(params, "", kEpsilon);
 *   solver.SetIndex(values_init.CreateIndex({'v'}));
 *   solver.Reset(values_init);
 *
 *   // Iterate to convergence
 *   sym::optimization_stats_t stats;
 *   bool should_early_exit = false;
 *   while (!should_early_exit) {
 *     should_early_exit = solver.Iterate(residual_func, &stats);
 *   }
 *
 *   // Get the best values
 *   sym::Valuesd values_final = solver.GetBestValues();
 *
 * The theory:
 *
 *   We start with a nonlinear vector-valued error function that defines an error residual for
 *   which we want to minimize the squared norm. The residual is dimension M, the state is N.
 *     residual = f(x)
 *
 *   Define a least squares cost function as the squared norm of the residual:
 *     e(x) = 0.5 * ||f(x)||**2 = 0.5 * f(x).T * f(x)
 *
 *   Take the first order taylor expansion for x around the linearization point x0:
 *     f(x) = f(x0) + f'(x0) * (x - x0) + ...
 *
 *   Plug in to the cost function to get a quadratic:
 *     e(x) ~= 0.5 * (x - x0).T * f'(x0).T * f'(x0) * (x - x0) + f(x0).T * f'(x0) * (x - x0)
 *             + 0.5 * f(x0).T * f(x0)
 *
 *   Take derivative with respect to x:
 *     e'(x) = f'(x0).T * f'(x0) * (x - x0) + f'(x0).T * f(x0)
 *
 *   Set to zero to find the minimum value of the quadratic (paraboloid):
 *     0 = f'(x0).T * f'(x0) * (x - x0) + f'(x0).T * f(x0)
 *     (x - x0) = - inv(f'(x0).T * f'(x0)) * f'(x0).T * f(x0)
 *     x = x0 - inv(f'(x0).T * f'(x0)) * f'(x0).T * f(x0)
 *
 *   Another way to write this is to create some helpful shorthand:
 *     f'(x0) --> jacobian or J (shape = MxN)
 *     f (x0) --> bias or b     (shape = Mx1)
 *     x - x0 --> dx            (shape = Nx1)
 *
 *   Rederiving the Gauss-Newton solution:
 *     e(x) ~= 0.5 * dx.T * J.T * J * dx + b.T * J * dx + 0.5 * b.T * b
 *     e'(x) = J.T * J * dx + J.T * b
 *     x = x0 - inv(J.T * J) * J.T * b
 *
 *   A couple more common names:
 *     f'(x0).T * f'(x0) = J.T * J --> hessian approximation or H (shape = NxN)
 *     f'(x0).T * f (x0) = J.T * b --> right hand side or rhs     (shape = Nx1)
 *
 *   So the iteration loop for optimization is:
 *     J, b = linearize(f, x0)
 *     dx = -inv(J.T * J) * J.T * b
 *     x_new = x0 + dx
 */
template <typename ScalarType,
          typename LinearSolverType = math::SparseCholeskySolver<Eigen::SparseMatrix<ScalarType>>>
class LevenbergMarquardtSolver {
 public:
  using Scalar = ScalarType;
  using LinearSolver = LinearSolverType;
  using StateType = internal::LevenbergMarquardtState<Scalar>;

  // Function that evaluates the objective function and produces a quadratic approximation of
  // it by linearizing a least-squares residual.
  using LinearizeFunc = std::function<void(const Values<Scalar>&, Linearization<Scalar>* const)>;

  LevenbergMarquardtSolver(const optimizer_params_t& p, const std::string& id, const Scalar epsilon)
      : p_(p), id_(id), epsilon_(epsilon) {}

  void SetIndex(const index_t& index) {
    index_ = index;
  }

  // Create an initial state to start a new optimization.
  void Reset(const Values<Scalar>& values) {
    // Should have called SetIndex already
    SYM_ASSERT(!index_.entries.empty());

    current_lambda_ = p_.initial_lambda;
    iteration_ = -1;

    ResetState(values);
  }

  // Reset the state values, such as if the cost function changes and linearizations are invalid.
  // Resets the values for optimization, but doesn't reset lambda or the number of iterations.
  void ResetState(const Values<Scalar>& values) {
    // Should have called SetIndex already
    SYM_ASSERT(!index_.entries.empty());

    max_diagonal_ = boost::none;
    last_update_ = boost::none;

    state_.Reset(values);
  }

  const optimizer_params_t& Params() const {
    return p_;
  }

  void UpdateParams(const optimizer_params_t& p);

  // Run one iteration of the optimization. Returns true if the optimization should early exit.
  bool Iterate(const LinearizeFunc& func, optimization_stats_t* const stats,
               const bool debug_stats = false);

  const Values<Scalar>& GetBestValues() const {
    SYM_ASSERT(state_.BestIsValid());
    return state_.Best().values;
  }

  void ComputeCovarianceAtBest(sym::MatrixX<Scalar>* const covariance);

 private:
  Eigen::SparseMatrix<Scalar> DampHessian(const Eigen::SparseMatrix<Scalar>& hessian_lower,
                                          boost::optional<VectorX<Scalar>>* const max_diagonal,
                                          const Scalar lambda) const;

  void CheckHessianDiagonal(const Eigen::SparseMatrix<Scalar>& hessian_lower_damped);

  void PopulateIterationStats(optimization_iteration_t* const iteration_stats,
                              const StateType& state, const Scalar new_error,
                              const Scalar relative_reduction, const bool debug_stats) const;

  void Update(const Values<Scalar>& values, const index_t& index, const VectorX<Scalar>& update,
              Values<Scalar>* const updated_values) const;

  optimizer_params_t p_;
  std::string id_;

  Scalar epsilon_;

  // State blocks for the optimizer
  StateType state_;

  LinearSolver linear_solver_{};
  bool solver_analyzed_{false};

  // Current elementwise max of the Hessian diagonal across all iterations, used for damping
  boost::optional<VectorX<Scalar>> max_diagonal_;

  // Current value of the damping parameter lambda
  Scalar current_lambda_;

  // Previous update vector, used to decide if we should take an uphill "bold" step
  boost::optional<VectorX<Scalar>> last_update_;

  int iteration_{-1};

  // Working storage to avoid reallocation
  VectorX<Scalar> update_;
  Eigen::SparseMatrix<Scalar> H_damped_;
  Eigen::Array<bool, Eigen::Dynamic, 1> zero_diagonal_;
  std::vector<int> zero_diagonal_indices_;

  // Index for the associated values, used for values.Update or Retract
  index_t index_{};
};

}  // namespace sym

#include "./levenberg_marquardt_solver.tcc"
