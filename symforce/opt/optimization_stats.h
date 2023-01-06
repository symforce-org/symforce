/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <lcmtypes/sym/optimization_iteration_t.hpp>
#include <lcmtypes/sym/optimization_stats_t.hpp>

#include "./linearization.h"
#include "./optional.h"

namespace sym {

// Debug stats for a full optimization run
template <typename Scalar>
struct OptimizationStats {
  std::vector<optimization_iteration_t> iterations;

  // Index into iterations of the best iteration (containing the optimal Values)
  int32_t best_index{0};

  // Did the optimization early exit? (either because it converged, or because it could not find a
  // good step)
  bool early_exited{false};

  optional<Linearization<Scalar>> best_linearization{};

  sparse_matrix_structure_t jacobian_sparsity;
  Eigen::VectorXi linear_solver_ordering;
  sparse_matrix_structure_t cholesky_factor_sparsity;

  optimization_stats_t GetLcmType() const {
    return optimization_stats_t(iterations, best_index, early_exited, jacobian_sparsity,
                                linear_solver_ordering, cholesky_factor_sparsity);
  }

  // Reset the optimization stats
  // Does _not_ cause reallocation, except for things in debug stats
  void Reset(const size_t num_iterations) {
    iterations.clear();
    iterations.reserve(num_iterations);

    best_index = {};
    early_exited = {};
    best_linearization = {};
    jacobian_sparsity = {};
    linear_solver_ordering = {};
    cholesky_factor_sparsity = {};
  }
};

// Shorthand instantiations
using OptimizationStatsd = OptimizationStats<double>;
using OptimizationStatsf = OptimizationStats<float>;

}  // namespace sym
