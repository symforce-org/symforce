/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/SparseCore>

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
  /// populate_best_linearization = true
  optional<Linearization<MatrixType>> best_linearization{};

  /// The sparsity pattern of the problem jacobian
  ///
  /// Only filled if Optimizer created with debug_stats = true and include_jacobians = true,
  /// otherwise default constructed.
  ///
  /// If using a dense linearization, only the shape field will be filled.
  sparse_matrix_structure_t jacobian_sparsity{};

  /// The permutation used by the linear solver
  ///
  /// Only filled if using an Optimizer created with debug_stats = true and a linear solver that
  /// exposes Permutation() (such as the default SparseCholeskySolver).  Otherwise, will be default
  /// constructed.
  Eigen::VectorXi linear_solver_ordering{};

  /// The sparsity pattern of the cholesky factor L
  ///
  /// Only filled if using an Optimizer created with debug_stats = true and a linear solver that
  /// exposes L() (such as the default SparseCholeskySolver).  Otherwise, will be default
  /// constructed.
  sparse_matrix_structure_t cholesky_factor_sparsity{};

  optimization_stats_t GetLcmType() const {
    return optimization_stats_t(iterations, best_index, status, failure_reason, jacobian_sparsity,
                                linear_solver_ordering, cholesky_factor_sparsity);
  }

  /// Get a view of the Jacobian at a particular iteration
  ///
  /// The lifetime of the result is tied to the lifetime of the OptimizationStats object
  auto JacobianView(const optimization_iteration_t& iteration) const {
    SYM_ASSERT(
        jacobian_sparsity.shape.size() == 2,
        "Jacobian sparsity is empty, did you set debug_stats = true and include_jacobians = true?");
    return MatrixViewFromSparseStructure<MatrixType>(jacobian_sparsity, iteration.jacobian_values);
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
