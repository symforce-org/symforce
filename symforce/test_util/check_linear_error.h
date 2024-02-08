/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <symforce/opt/optimization_stats.h>

namespace sym {

/// Check that the linear error in the optimization stats matches the linear error computed from the
/// Jacobian
///
/// Use from tests, adds Catch2 checks.  Requires debug_stats and include_jacobians to be turned on.
template <typename OptimizationStats>
void CheckLinearError(const OptimizationStats& stats) {
  const sym::optimization_iteration_t* last_accepted_iteration = &stats.iterations.at(0);
  for (int i = 1; i < static_cast<int>(stats.iterations.size()); ++i) {
    const auto& iteration = stats.iterations.at(i);

    const auto J = stats.JacobianView(*last_accepted_iteration).template cast<double>().eval();
    CAPTURE(J);
    const Eigen::VectorXd new_residual = J * iteration.update + last_accepted_iteration->residual;

    const auto new_cost = 0.5 * new_residual.squaredNorm();

    CAPTURE(i, last_accepted_iteration->iteration - 1);
    CHECK(new_cost == Catch::Approx(iteration.new_error_linear).epsilon(1e-6));

    if (iteration.update_accepted) {
      last_accepted_iteration = &iteration;
    }
  }
}

}  // namespace sym
