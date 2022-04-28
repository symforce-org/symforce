/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "./levenberg_marquardt_solver.h"
#include "./tic_toc.h"
#include "./util.h"

namespace sym {

// ----------------------------------------------------------------------------
// Private methods
// ----------------------------------------------------------------------------

template <typename ScalarType, typename LinearSolverType>
Eigen::SparseMatrix<ScalarType> LevenbergMarquardtSolver<ScalarType, LinearSolverType>::DampHessian(
    const Eigen::SparseMatrix<Scalar>& hessian_lower, bool* const have_max_diagonal,
    VectorX<Scalar>* const max_diagonal, const Scalar lambda) const {
  SYM_TIME_SCOPE("LM<{}>: DampHessian", id_);
  Eigen::SparseMatrix<Scalar> H_damped = hessian_lower;

  if (p_.use_diagonal_damping) {
    if (p_.keep_max_diagonal_damping) {
      if (!*have_max_diagonal) {
        *max_diagonal = H_damped.diagonal();
        *max_diagonal = max_diagonal->cwiseMax(p_.diagonal_damping_min);
      } else {
        *max_diagonal = max_diagonal->cwiseMax(H_damped.diagonal());
      }

      *have_max_diagonal = true;

      H_damped.diagonal().array() += max_diagonal->array() * lambda;
    } else {
      H_damped.diagonal().array() += H_damped.diagonal().array() * lambda;
    }
  }

  if (p_.use_unit_damping) {
    H_damped.diagonal().array() += lambda;
  }

  return H_damped;
}

template <typename ScalarType, typename LinearSolverType>
void LevenbergMarquardtSolver<ScalarType, LinearSolverType>::CheckHessianDiagonal(
    const Eigen::SparseMatrix<Scalar>& hessian_lower_damped) {
  zero_diagonal_ = hessian_lower_damped.diagonal().array().abs() < epsilon_;

  // NOTE(aaron): We call this outside the condition so it's guaranteed to do the allocation on the
  // first iteration
  zero_diagonal_indices_.reserve(zero_diagonal_.rows());

  if (zero_diagonal_.any()) {
    zero_diagonal_indices_.clear();

    for (int i = 0; i < zero_diagonal_.rows(); i++) {
      if (zero_diagonal_(i)) {
        zero_diagonal_indices_.push_back(i);
      }
    }

    spdlog::warn("LM<{}> Zero on diagonal at indices: {}", id_, zero_diagonal_indices_);
  }
}

template <typename ScalarType, typename LinearSolverType>
void LevenbergMarquardtSolver<ScalarType, LinearSolverType>::PopulateIterationStats(
    optimization_iteration_t* const iteration_stats, const StateType& state_,
    const Scalar new_error, const Scalar relative_reduction, const bool debug_stats) const {
  SYM_TIME_SCOPE("LM<{}>: IterationStats", id_);

  iteration_stats->iteration = iteration_;
  iteration_stats->current_lambda = current_lambda_;

  iteration_stats->new_error = new_error;
  iteration_stats->relative_reduction = relative_reduction;

  {
    SYM_TIME_SCOPE("LM<{}>: IterationStats - LinearErrorFromValues", id_);
    iteration_stats->new_error_linear = state_.Init().GetLinearization().LinearError(update_);
  }

  if (p_.verbose) {
    SYM_TIME_SCOPE("LM<{}>: IterationStats - Print", id_);
    spdlog::info(
        "[iter {:4d}] lambda: {:.3e}, error prev/linear/new: {:.3f}/{:.3f}/{:.3f}, "
        "rel reduction: {:.5f}",
        iteration_stats->iteration, iteration_stats->current_lambda, state_.Init().Error(),
        iteration_stats->new_error_linear, iteration_stats->new_error,
        iteration_stats->relative_reduction);
  }

  if (debug_stats) {
    iteration_stats->values = state_.New().values.template Cast<double>().GetLcmType();
    const VectorX<Scalar> residual_vec = state_.New().GetLinearization().residual;
    iteration_stats->residual = residual_vec.template cast<float>();
    const VectorX<Scalar> jacobian_vec = state_.New().GetLinearization().JacobianValuesMap();
    iteration_stats->jacobian_values = jacobian_vec.template cast<float>();
  }
}

template <typename ScalarType, typename LinearSolverType>
void LevenbergMarquardtSolver<ScalarType, LinearSolverType>::Update(
    const Values<Scalar>& values, const index_t& index, const VectorX<Scalar>& update,
    Values<Scalar>* const updated_values) const {
  SYM_ASSERT(update.rows() == index.tangent_dim);

  if (updated_values->NumEntries() == 0) {
    // If the state_ blocks are empty the first time, copy in the full structure
    (*updated_values) = values;
  } else {
    // Otherwise just copy the keys being optimized
    updated_values->Update(index, values);
  }

  // Apply the update
  updated_values->Retract(index, update.data(), epsilon_);
}

// ----------------------------------------------------------------------------
// Public methods
// ----------------------------------------------------------------------------

template <typename ScalarType, typename LinearSolverType>
void LevenbergMarquardtSolver<ScalarType, LinearSolverType>::UpdateParams(
    const optimizer_params_t& p) {
  if (p_.verbose) {
    spdlog::info("LM<{}>: UPDATING OPTIMIZER PARAMS", id_);
  }
  p_ = p;
}

template <typename ScalarType, typename LinearSolverType>
bool LevenbergMarquardtSolver<ScalarType, LinearSolverType>::Iterate(
    const LinearizeFunc& func, OptimizationStats<Scalar>* const stats, const bool debug_stats) {
  SYM_TIME_SCOPE("LM<{}>::Iterate()", id_);
  SYM_ASSERT(stats != nullptr);

  // new -> init
  {
    SYM_TIME_SCOPE("LM<{}>: StateStep", id_);
    state_.Step();
    iteration_++;
  }

  if (!state_.Init().GetLinearization().IsInitialized()) {
    SYM_TIME_SCOPE("LM<{}>: EvaluateFirst", id_);
    state_.Init().Relinearize(func);
    state_.SetBestToInit();
  }

  // save the initial error state_ before optimizing
  if (iteration_ == 0) {
    SYM_TIME_SCOPE("LM<{}>: FirstIterationStats", id_);
    stats->iterations.emplace_back();
    optimization_iteration_t& iteration_stats = stats->iterations.back();
    iteration_stats.iteration = -1;
    iteration_stats.new_error = state_.Init().Error();
    iteration_stats.current_lambda = current_lambda_;

    if (debug_stats) {
      iteration_stats.values = state_.Init().values.template Cast<double>().GetLcmType();
      const VectorX<Scalar> residual_vec = state_.Init().GetLinearization().residual;
      iteration_stats.residual = residual_vec.template cast<float>();
      const VectorX<Scalar> jacobian_vec = state_.Init().GetLinearization().JacobianValuesMap();
      iteration_stats.jacobian_values = jacobian_vec.template cast<float>();
    }
  }

  // Analyze the sparsity pattern for efficient repeated factorization
  if (!solver_analyzed_) {
    // TODO(aaron): Do this with the ones linearization computed by the Linearizer
    SYM_TIME_SCOPE("LM<{}>: AnalyzePattern", id_);
    Eigen::SparseMatrix<Scalar> H_analyze = state_.Init().GetLinearization().hessian_lower;
    H_analyze.diagonal().array() = 1.0;  // Make sure the diagonal is nonzero for analysis
    linear_solver_.ComputeSymbolicSparsity(H_analyze);
    solver_analyzed_ = true;
  }

  // TODO(aaron): Get rid of this copy
  H_damped_ = DampHessian(state_.Init().GetLinearization().hessian_lower, &have_max_diagonal_,
                          &max_diagonal_, current_lambda_);

  CheckHessianDiagonal(H_damped_);

  {
    SYM_TIME_SCOPE("LM<{}>: SparseFactorize", id_);
    linear_solver_.Factorize(H_damped_);
  }

  {
    SYM_TIME_SCOPE("LM<{}>: SparseSolve", id_);
    update_ = linear_solver_.Solve(state_.Init().GetLinearization().rhs);
  }

  {
    SYM_TIME_SCOPE("LM<{}>: Update", id_);
    Update(state_.Init().values, index_, -update_, &state_.New().values);
  }

  {
    SYM_TIME_SCOPE("LM<{}>: linearization_func", id_);
    state_.New().Relinearize(func);
  }

  const Scalar new_error = state_.New().Error();
  const Scalar relative_reduction =
      (state_.Init().Error() - new_error) / (state_.Init().Error() + epsilon_);

  stats->iterations.emplace_back();
  optimization_iteration_t& iteration_stats = stats->iterations.back();
  PopulateIterationStats(&iteration_stats, state_, new_error, relative_reduction, debug_stats);

  if (!std::isfinite(new_error)) {
    spdlog::warn("LM<{}> Encountered non-finite error: {}", id_, new_error);
  }

  // Early exit if the reduction in error is too small.
  bool should_early_exit =
      (relative_reduction > 0) && (relative_reduction < p_.early_exit_min_reduction);

  {
    SYM_TIME_SCOPE("LM<{}>: accept_update bookkeeping", id_);
    bool accept_update = relative_reduction > 0;

    // NOTE(jack): Reference https://arxiv.org/abs/1201.5885
    Scalar update_angle_change = 0;
    if (p_.enable_bold_updates && have_last_update_ && !accept_update) {
      update_angle_change = last_update_.normalized().dot(update_.stableNormalized());

      accept_update = (Square(1 - update_angle_change) * new_error) <= state_.Best().Error();
    }

    // If we didn't accept the update and lambda is maxed out, just exit.
    should_early_exit |= (!accept_update && current_lambda_ >= p_.lambda_upper_bound);

    if (!accept_update) {
      current_lambda_ *= p_.lambda_up_factor;
      // swap state_ blocks so that the next iteration gets the same initial state_ as this one
      state_.SwapNewAndInit();
    } else {
      current_lambda_ *= p_.lambda_down_factor;
      have_last_update_ = true;
      last_update_ = update_;
      if (state_.New().Error() <= state_.Best().Error()) {
        state_.SetBestToNew();
        stats->best_index = stats->iterations.size() - 1;
      }
      // Ensure that we are not going to modify the Best state_ block in the next iteration
      state_.SetInitToNotBest();
    }

    // Clip lambda to bounds
    current_lambda_ = Clamp(current_lambda_, p_.lambda_lower_bound, p_.lambda_upper_bound);

    // Finish populating iteration_stats
    iteration_stats.update_angle_change = update_angle_change;
    iteration_stats.update_accepted = accept_update;
  }

  return should_early_exit;
}

template <typename ScalarType, typename LinearSolverType>
void LevenbergMarquardtSolver<ScalarType, LinearSolverType>::ComputeCovariance(
    const Eigen::SparseMatrix<Scalar>& hessian_lower, MatrixX<Scalar>* const covariance) {
  SYM_TIME_SCOPE("LM<{}>: ComputeCovariance()", id_);

  H_damped_ = hessian_lower;
  H_damped_.diagonal().array() += epsilon_;

  // TODO(hayk, aaron): This solver assumes a dense RHS, should add support for a sparse RHS
  linear_solver_.Factorize(H_damped_);
  *covariance = MatrixX<Scalar>::Identity(H_damped_.rows(), H_damped_.rows());
  linear_solver_.SolveInPlace(covariance);
}

// ----------------------------------------------------------------------------
// Shorthand instantiations
// ----------------------------------------------------------------------------

using LevenbergMarquardtSolverd = LevenbergMarquardtSolver<double>;
using LevenbergMarquardtSolverf = LevenbergMarquardtSolver<float>;

}  // namespace sym
