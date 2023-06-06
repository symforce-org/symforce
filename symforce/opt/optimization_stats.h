/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Sparse>

#include <lcmtypes/sym/optimization_iteration_t.hpp>
#include <lcmtypes/sym/optimization_stats_t.hpp>

#include "./linearization.h"
#include "./optional.h"

namespace sym {

/**
 * Debug stats for a full optimization run
 */
template <typename MatrixType>
struct OptimizationStats {
  std::vector<optimization_iteration_t> iterations;

  /// Index into iterations of the best iteration (containing the optimal Values)
  int32_t best_index{0};

  /// What was the result of the optimization?
  optimization_status_t status{};

  /// If status == FAILED, why?  This should be cast to the Optimizer::FailureReason enum for the
  /// nonlinear solver you used.
  int32_t failure_reason{};

  /// The linearization at best_index (at optimized_values), filled out if
  /// populate_best_linearization=true
  optional<Linearization<MatrixType>> best_linearization{};

  /// The sparsity pattern of the problem jacobian
  ///
  /// Only filled if using sparse linear solver and Optimizer created with debug_stats = true.
  /// If not filled, row_indices field of sparse_matrix_structure_t and linear_solver_ordering
  /// will have size() = 0.
  sparse_matrix_structure_t jacobian_sparsity;

  /// The permutation used by the linear solver
  ///
  /// Only filled if using sparse linear solver and Optimizer created with debug_stats = true.
  /// If not filled, row_indices field of sparse_matrix_structure_t and linear_solver_ordering
  /// will have size() = 0.
  Eigen::VectorXi linear_solver_ordering;

  /// The sparsity pattern of the cholesky factor
  ///
  /// Only filled if using sparse linear solver and Optimizer created with debug_stats = true.
  /// If not filled, row_indices field of sparse_matrix_structure_t and linear_solver_ordering
  /// will have size() = 0.
  sparse_matrix_structure_t cholesky_factor_sparsity;

  optimization_stats_t GetLcmType() const {
    return optimization_stats_t(iterations, best_index, status, failure_reason, jacobian_sparsity,
                                linear_solver_ordering, cholesky_factor_sparsity);
  }

  /**
   * Reset the optimization stats
   *
   * Does _not_ cause reallocation, except for things in debug stats
   */
  void Reset(const size_t num_iterations) {
    iterations.clear();
    iterations.reserve(num_iterations);

    best_index = {};
    status = {};
    failure_reason = {};
    best_linearization = {};
    jacobian_sparsity = {};
    linear_solver_ordering = {};
    cholesky_factor_sparsity = {};
  }
};

// Shorthand instantiations
template <typename Scalar>
using SparseOptimizationStats = OptimizationStats<Eigen::SparseMatrix<Scalar>>;
template <typename Scalar>
using DenseOptimizationStats = OptimizationStats<MatrixX<Scalar>>;
using OptimizationStatsd = SparseOptimizationStats<double>;
using OptimizationStatsf = SparseOptimizationStats<float>;

}  // namespace sym
