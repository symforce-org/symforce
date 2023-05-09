/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */
#include <array>
#include <iostream>

#include <Eigen/Dense>
#include <spdlog/spdlog.h>

#include <lcmtypes/sym/optimization_iteration_t.hpp>

#include <sym/ops/lie_group_ops.h>
#include <sym/pose2.h>
#include <sym/util/epsilon.h>
#include <symforce/opt/assert.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/key.h>
#include <symforce/opt/optimization_stats.h>
#include <symforce/opt/optimizer.h>

#include "./gen/bearing_factor.h"
#include "./gen/odometry_factor.h"

namespace robot_2d_localization {

void RunLocalization() {
  const int num_poses = 3;
  const int num_landmarks = 3;

  std::vector<sym::Factor<double>> factors;

  // Bearing factors
  for (int i = 0; i < num_poses; ++i) {
    for (int j = 0; j < num_landmarks; ++j) {
      factors.push_back(sym::Factor<double>::Hessian(
          sym::BearingFactor<double>, {{'P', i}, {'L', j}, {'a', i, j}, {'e'}},  // keys
          {{'P', i}}                                                             // keys to optimize
          ));
    }
  }

  // Odometry factors
  for (int i = 0; i < num_poses - 1; ++i) {
    factors.push_back(sym::Factor<double>::Hessian(
        sym::OdometryFactor<double>, {{'P', i}, {'P', i + 1}, {'d', i}, {'e'}},  // keys
        {{'P', i}, {'P', i + 1}}                                                 // keys to optimize
        ));
  }

  auto params = sym::DefaultOptimizerParams();
  params.verbose = true;
  sym::Optimizer<double> optimizer(params, factors, sym::kDefaultEpsilon<double>);

  sym::Values<double> values;
  for (int i = 0; i < num_poses; ++i) {
    values.Set({'P', i}, sym::Pose2d::Identity());
  }

  // Set additional values
  values.Set({'L', 0}, Eigen::Vector2d(-2, 2));
  values.Set({'L', 1}, Eigen::Vector2d(1, -3));
  values.Set({'L', 2}, Eigen::Vector2d(5, 2));
  values.Set({'d', 0}, 1.7);
  values.Set({'d', 1}, 1.4);
  const std::array<std::array<double, 3>, 3> angles = {
      {{55, 245, -35}, {95, 220, -20}, {125, 220, -20}}};
  for (int i = 0; i < static_cast<int>(angles.size()); ++i) {
    for (int j = 0; j < static_cast<int>(angles[0].size()); ++j) {
      values.Set({'a', i, j}, angles[i][j] * M_PI / 180);
    }
  }
  values.Set('e', sym::kDefaultEpsilond);

  // Optimize!
  const auto stats = optimizer.Optimize(values);

  spdlog::debug("Optimized values: {}", values);

  // Check output
  const auto& iteration_stats = stats.iterations;
  const auto& first_iter = iteration_stats.front();
  const auto& best_iter = iteration_stats[stats.best_index];

  SYM_ASSERT(iteration_stats.size() == 9);
  SYM_ASSERT(6.39635 < first_iter.new_error && first_iter.new_error < 6.39637);
  SYM_ASSERT(best_iter.new_error < 0.00022003);
  SYM_ASSERT(stats.status == sym::optimization_status_t::SUCCESS);

  const sym::Pose2d expected_p0({0.477063, 0.878869, -0.583038, -0.824491});
  const sym::Pose2d expected_p1({0.65425, 0.756279, 1.01671, -0.238356});
  const sym::Pose2d expected_p2({0.779849, 0.625967, 1.79785, 0.920551});

  SYM_ASSERT(sym::IsClose(expected_p0, values.At<sym::Pose2d>({'P', 0}), 1e-6));
  SYM_ASSERT(sym::IsClose(expected_p1, values.At<sym::Pose2d>({'P', 1}), 1e-6));
  SYM_ASSERT(sym::IsClose(expected_p2, values.At<sym::Pose2d>({'P', 2}), 1e-6));
}

}  // namespace robot_2d_localization
