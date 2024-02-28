/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "./assert.h"
#include "./levenberg_marquardt_solver.h"
#include "./tic_toc.h"
#include "./util.h"

namespace sym {

// ----------------------------------------------------------------------------
// Private methods
// ----------------------------------------------------------------------------

template <typename ScalarType, typename LinearSolverType>
void LevenbergMarquardtSolver<ScalarType, LinearSolverType>::DampHessian(
    MatrixType& hessian_lower, bool& have_max_diagonal, VectorX<Scalar>& max_diagonal,
    const Scalar lambda, VectorX<Scalar>& damping_vector,
    VectorX<Scalar>& undamped_diagonal) const {
  SYM_TIME_SCOPE("LM<{}>: DampHessian", id_);

  undamped_diagonal = hessian_lower.diagonal();

  if (p_.use_diagonal_damping) {
    if (p_.keep_max_diagonal_damping) {
      if (!have_max_diagonal) {
        max_diagonal = undamped_diagonal.cwiseMax(p_.diagonal_damping_min);
      } else {
        max_diagonal = max_diagonal.cwiseMax(undamped_diagonal);
      }

      have_max_diagonal = true;

      damping_vector = max_diagonal * lambda;
    } else {
      damping_vector = undamped_diagonal.cwiseMax(p_.diagonal_damping_min) * lambda;
    }
  } else {
    damping_vector = VectorX<Scalar>::Zero(hessian_lower.rows());
  }

  if (p_.use_unit_damping) {
    damping_vector.array() += lambda;
  }

  hessian_lower.diagonal() += damping_vector;
}

template <typename ScalarType, typename LinearSolverType>
void LevenbergMarquardtSolver<ScalarType, LinearSolverType>::CheckHessianDiagonal(
    const MatrixType& hessian_lower_damped, const Scalar lambda) {
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

    constexpr int max_indices_to_show{15};
    if (zero_diagonal_indices_.size() > max_indices_to_show) {
      spdlog::warn(
          "LM<{}> Zero on diagonal after damping (with lambda = {:.2e}, epsilon = {:.2e}) at "
          "indices: [{}, ... ({} omitted)]",
          id_, lambda, epsilon_,
          fmt::join(zero_diagonal_indices_.cbegin(),
                    zero_diagonal_indices_.cbegin() + max_indices_to_show, ", "),
          zero_diagonal_indices_.size() - max_indices_to_show);
    } else {
      spdlog::warn(
          "LM<{}> Zero on diagonal after damping (with lambda = {:.2e}, epsilon = {:.2e}) at "
          "indices: {}",
          id_, lambda, epsilon_, zero_diagonal_indices_);
    }
  }
}

template <typename ScalarType, typename LinearSolverType>
void LevenbergMarquardtSolver<ScalarType, LinearSolverType>::PopulateIterationStats(
    optimization_iteration_t& iteration_stats, const StateType& state, const Scalar new_error,
    const Scalar new_error_linear, const Scalar relative_reduction, const Scalar gain_ratio) const {
  SYM_TIME_SCOPE("LM<{}>: IterationStats", id_);

  iteration_stats.iteration = iteration_;
  iteration_stats.current_lambda = current_lambda_;

  iteration_stats.new_error = new_error;
  iteration_stats.new_error_linear = new_error_linear;
  iteration_stats.relative_reduction = relative_reduction;

  if (p_.verbose) {
    SYM_TIME_SCOPE("LM<{}>: IterationStats - Print", id_);
    spdlog::info(
        "LM<{}> [iter {:4d}] lambda: {:.3e}, error prev/linear/new: {:.3e}/{:.3e}/{:.3e}, "
        "rel reduction: {:.5e}, gain ratio: {:.5e}",
        id_, iteration_stats.iteration, iteration_stats.current_lambda, state.Init().Error(),
        iteration_stats.new_error_linear, iteration_stats.new_error,
        iteration_stats.relative_reduction, gain_ratio);
  }

  if (p_.debug_stats) {
    iteration_stats.update = update_.template cast<double>();
    iteration_stats.values = state.New().values.template Cast<double>().GetLcmType();
    const VectorX<Scalar> residual_vec = state.New().GetLinearization().residual;
    iteration_stats.residual = residual_vec.template cast<double>();
    const MatrixX<Scalar> jacobian_vec = JacobianValues(state.New().GetLinearization().jacobian);
    iteration_stats.jacobian_values = jacobian_vec.template cast<double>();
  }
}

template <typename ScalarType, typename LinearSolverType>
void LevenbergMarquardtSolver<ScalarType, LinearSolverType>::Update(
    const Values<Scalar>& values, const index_t& index, const VectorX<Scalar>& update,
    Values<Scalar>& updated_values) const {
  SYM_ASSERT(update.rows() == index.tangent_dim);

  if (updated_values.NumEntries() == 0) {
    // If the state_ blocks are empty the first time, copy in the full structure
    updated_values = values;
  } else {
    // Otherwise just copy the keys being optimized
    updated_values.Update(index, values);
  }

  // Apply the update
  updated_values.Retract(index, update.data(), epsilon_);
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
optional<std::pair<optimization_status_t, levenberg_marquardt_solver_failure_reason_t>>
LevenbergMarquardtSolver<ScalarType, LinearSolverType>::Iterate(
    const LinearizeFunc& func, OptimizationStats<MatrixType>& stats) {
  SYM_TIME_SCOPE("LM<{}>::Iterate()", id_);

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
    stats.iterations.emplace_back();
    optimization_iteration_t& iteration_stats = stats.iterations.back();
    iteration_stats.iteration = -1;
    iteration_stats.new_error = state_.Init().Error();
    iteration_stats.current_lambda = current_lambda_;

    if (p_.debug_stats) {
      iteration_stats.values = state_.Init().values.template Cast<double>().GetLcmType();
      const VectorX<Scalar> residual_vec = state_.Init().GetLinearization().residual;
      iteration_stats.residual = residual_vec.template cast<double>();
      const MatrixX<Scalar> jacobian_vec =
          JacobianValues(state_.Init().GetLinearization().jacobian);
      iteration_stats.jacobian_values = jacobian_vec.template cast<double>();
    }

    if (!std::isfinite(state_.Init().Error())) {
      spdlog::warn("LM<{}> Encountered non-finite initial error: {}", id_, state_.Init().Error());
      if (!p_.debug_checks) {
        spdlog::warn("LM<{}> Turn on debug_checks to see which factor is causing this", id_);
      }
      return {{optimization_status_t::FAILED, FailureReason::INITIAL_ERROR_NOT_FINITE}};
    }
  }

  // Analyze the sparsity pattern for efficient repeated factorization
  if (!solver_analyzed_) {
    SYM_TIME_SCOPE("LM<{}>: AnalyzePattern", id_);
    linear_solver_.AnalyzeSparsityPattern(state_.Init().GetLinearization().hessian_lower);
    solver_analyzed_ = true;
  }

  DampHessian(state_.Init().GetLinearization().hessian_lower, have_max_diagonal_, max_diagonal_,
              current_lambda_, damping_vector_, undamped_diagonal_);

  CheckHessianDiagonal(state_.Init().GetLinearization().hessian_lower, current_lambda_);

  {
    SYM_TIME_SCOPE("LM<{}>: SparseFactorize", id_);
    const bool success = linear_solver_.Factorize(state_.Init().GetLinearization().hessian_lower);
    // TODO(brad): Instead try recovering from this (ultimately by increasing lambda).
    SYM_ASSERT(success, "Internal Error: Damped hessian factorization failed");

    // NOTE(aaron): This has to happen after the first factorize, since L_inner is not filled out
    // by ComputeSymbolicSparsity.  The linear_solver may return an empty result for either of
    // these, so the only way to know we haven't filled it out yet is the iteration number.
    if (p_.debug_stats && iteration_ == 0) {
      stats.linear_solver_ordering = linear_solver_.Permutation().indices();
      stats.cholesky_factor_sparsity = GetSparseStructure(linear_solver_.L());
    }
  }

  {
    SYM_TIME_SCOPE("LM<{}>: SparseSolve", id_);
    update_ = -linear_solver_.Solve(state_.Init().GetLinearization().rhs);
  }

  {
    SYM_TIME_SCOPE("LM<{}>: ResetHessianDiagonal", id_);
    state_.Init().GetLinearization().hessian_lower.diagonal() = undamped_diagonal_;
  }

  if (p_.debug_checks && !update_.array().isFinite().all()) {
    spdlog::warn("LM<{}> Non-finite update: {}", id_, update_.transpose());
  }

  {
    SYM_TIME_SCOPE("LM<{}>: Update", id_);
    Update(state_.Init().values, index_, update_, state_.New().values);
  }

  state_.New().Relinearize(func);

  const Scalar new_error = state_.New().Error();
  const Scalar relative_reduction =
      (state_.Init().Error() - new_error) / (state_.Init().Error() + epsilon_);

  const Scalar new_error_linear = [this] {
    SYM_TIME_SCOPE("LM<{}>: LinearErrorFromValues", id_);
    return state_.Init().Error() +
           state_.Init().GetLinearization().LinearDeltaError(update_, damping_vector_);
  }();

  const Scalar gain_ratio =
      (state_.Init().Error() - new_error) / (state_.Init().Error() - new_error_linear);

  stats.iterations.emplace_back();
  optimization_iteration_t& iteration_stats = stats.iterations.back();
  PopulateIterationStats(iteration_stats, state_, new_error, new_error_linear, relative_reduction,
                         gain_ratio);

  if (!std::isfinite(new_error)) {
    spdlog::warn("LM<{}> Encountered non-finite error: {}", id_, new_error);
    if (!p_.debug_checks) {
      spdlog::warn("LM<{}> Turn on debug_checks to see which factor is causing this", id_);
    }
  }

  optional<std::pair<optimization_status_t, FailureReason>> status{};

  // Early exit if the reduction in error is too small.
  if (relative_reduction > -p_.early_exit_min_reduction / 10 &&
      relative_reduction < p_.early_exit_min_reduction) {
    status = {optimization_status_t::SUCCESS, {}};
  }

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
    if (!accept_update && current_lambda_ >= p_.lambda_upper_bound) {
      status = {optimization_status_t::FAILED, FailureReason::LAMBDA_OUT_OF_BOUNDS};
    }

    if (!accept_update) {
      switch (p_.lambda_update_type.value) {
        case lambda_update_type_t::INVALID:
          SYM_ASSERT(false, "Invalid lambda update type");
        case lambda_update_type_t::STATIC:
          current_lambda_ *= p_.lambda_up_factor;
          break;
        case lambda_update_type_t::DYNAMIC:
          current_lambda_ *= current_nu_;
          current_nu_ *= 2;
          break;
      }

      // swap state_ blocks so that the next iteration gets the same initial state_ as this one
      state_.SwapNewAndInit();
    } else {
      switch (p_.lambda_update_type.value) {
        case lambda_update_type_t::INVALID:
          SYM_ASSERT(false, "Invalid lambda update type");
        case lambda_update_type_t::STATIC:
          current_lambda_ *= p_.lambda_down_factor;
          break;
        case lambda_update_type_t::DYNAMIC:
          current_lambda_ *=
              std::max(Scalar{1} / static_cast<Scalar>(p_.dynamic_lambda_update_gamma),
                       Scalar{1} - (static_cast<Scalar>(p_.dynamic_lambda_update_beta) - 1) *
                                       std::pow(2 * gain_ratio - 1,
                                                static_cast<Scalar>(p_.dynamic_lambda_update_p)));
          current_nu_ = 2;
          break;
      }

      have_last_update_ = true;
      last_update_ = update_;
      if (state_.New().Error() <= state_.Best().Error()) {
        state_.SetBestToNew();
        stats.best_index = stats.iterations.size() - 1;
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

  return status;
}

template <typename ScalarType, typename LinearSolverType>
void LevenbergMarquardtSolver<ScalarType, LinearSolverType>::ComputeCovariance(
    const MatrixType& hessian_lower, MatrixX<Scalar>& covariance) {
  SYM_TIME_SCOPE("LM<{}>: ComputeCovariance()", id_);

  // TODO(hayk, aaron): This solver assumes a dense RHS, should add support for a sparse RHS
  const bool success = linear_solver_.Factorize(hessian_lower);
  // TODO(brad): Instead try recovering from this by damping?
  SYM_ASSERT(success, "Internal Error: damped hessian factorization failed");
  covariance = MatrixX<Scalar>::Identity(hessian_lower.rows(), hessian_lower.rows());
  linear_solver_.SolveInPlace(covariance);
}

// ----------------------------------------------------------------------------
// Shorthand instantiations
// ----------------------------------------------------------------------------

using LevenbergMarquardtSolverd = LevenbergMarquardtSolver<double>;
using LevenbergMarquardtSolverf = LevenbergMarquardtSolver<float>;

}  // namespace sym
