/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

///
/// Run with:
///
///     build/bin/benchmarks/robot_3d_localization_benchmark
///
/// See run_benchmarks.py for more information
///

#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <spdlog/spdlog.h>

#include <sym/rot3.h>
#include <symforce/examples/robot_3d_localization/common.h>
#include <symforce/examples/robot_3d_localization/gen/keys.h>
#include <symforce/examples/robot_3d_localization/gen/linearization.h>
#include <symforce/examples/robot_3d_localization/gen/matching_factor.h>
#include <symforce/examples/robot_3d_localization/gen/measurements.h>
#include <symforce/examples/robot_3d_localization/gen/odometry_factor.h>
#include <symforce/examples/robot_3d_localization/run_dynamic_size.h>
#include <symforce/examples/robot_3d_localization/run_fixed_size.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>
#include <symforce/opt/tic_toc.h>

#include "./robot_3d_localization_ceres.h"
#include "./robot_3d_localization_gtsam.h"

using namespace robot_3d_localization;

TEMPLATE_TEST_CASE("sym_dynamic_linearize", "", double, float) {
  using Scalar = TestType;

  sym::Values<Scalar> values = BuildValues<Scalar>(kNumPoses, kNumLandmarks);

  // Create and set up Optimizer
  const std::vector<sym::Factor<Scalar>> factors =
      BuildDynamicFactors<Scalar>(kNumPoses, kNumLandmarks);

  sym::Optimizer<Scalar> optimizer(RobotLocalizationOptimizerParams(), factors,
                                   sym::kDefaultEpsilon<Scalar>, "sym_dynamic_linearize");

  sym::Linearizer<Scalar>& linearizer = optimizer.Linearizer();
  sym::SparseLinearization<Scalar> linearization;

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  // Linearize
  {
    SYM_TIME_SCOPE("sym_dynamic_{}/linearize", typeid(Scalar).name());
    for (int i = 0; i < 1000; i++) {
      linearizer.Relinearize(values, linearization);
    }
  }
}

TEMPLATE_TEST_CASE("sym_dynamic_iterate", "", double, float) {
  using Scalar = TestType;

  sym::Values<Scalar> values = BuildValues<Scalar>(kNumPoses, kNumLandmarks);

  // Create and set up Optimizer
  const std::vector<sym::Factor<Scalar>> factors =
      BuildDynamicFactors<Scalar>(kNumPoses, kNumLandmarks);

  sym::Optimizer<Scalar> optimizer(RobotLocalizationOptimizerParams(), factors,
                                   sym::kDefaultEpsilon<Scalar>, "sym_dynamic_iterate");

  typename sym::Optimizer<Scalar>::Stats stats;

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  // Iterate
  {
    SYM_TIME_SCOPE("sym_dynamic_{}/iterate", typeid(Scalar).name());
    for (int i = 0; i < 1000; i++) {
      optimizer.Optimize(values, /* num_iterations */ 1, /* populate_best_linearization */ false,
                         stats);
    }
  }
}

TEMPLATE_TEST_CASE("sym_fixed_linearize", "", double, float) {
  using Scalar = TestType;

  sym::Values<Scalar> values = BuildValues<Scalar>(kNumPoses, kNumLandmarks);

  // Create and set up Optimizer
  const std::vector<sym::Factor<Scalar>> factors = {BuildFixedFactor<Scalar>()};

  sym::Optimizer<Scalar> optimizer(RobotLocalizationOptimizerParams(), factors,
                                   sym::kDefaultEpsilon<Scalar>, "sym_fixed_linearize");

  sym::Linearizer<Scalar>& linearizer = optimizer.Linearizer();
  sym::SparseLinearization<Scalar> linearization;

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  // Linearize
  {
    SYM_TIME_SCOPE("sym_fixed_{}/linearize", typeid(Scalar).name());
    for (int i = 0; i < 1000; i++) {
      linearizer.Relinearize(values, linearization);
    }
  }
}

TEMPLATE_TEST_CASE("sym_fixed_iterate", "", double, float) {
  using Scalar = TestType;

  sym::Values<Scalar> values = BuildValues<Scalar>(kNumPoses, kNumLandmarks);

  // Create and set up Optimizer
  const std::vector<sym::Factor<Scalar>> factors = {BuildFixedFactor<Scalar>()};

  sym::Optimizer<Scalar> optimizer(RobotLocalizationOptimizerParams(), factors,
                                   sym::kDefaultEpsilon<Scalar>, "sym_fixed_iterate");

  typename sym::Optimizer<Scalar>::Stats stats;

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  // Iterate
  {
    SYM_TIME_SCOPE("sym_fixed_{}/iterate", typeid(Scalar).name());
    for (int i = 0; i < 1000; i++) {
      optimizer.Optimize(values, /* num_iterations */ 1, /* populate_best_linearization */ false,
                         stats);
    }
  }
}

TEST_CASE("gtsam_linearize") {
  using namespace gtsam;

  ExpressionFactorGraph graph = BuildGtsamFactors();

  // Initial values.
  Values initial;
  for (int i = 0; i < kNumPoses; ++i) {
    initial.insert(i, Pose3::identity());
  }

  // Build optimizer
  LevenbergMarquardtParams params{};
  params.setVerbosityLM("SILENT");

  LevenbergMarquardtOptimizer optimizer(graph, initial, params);
  // Values result = optimizer.optimize();
  // result.print("Final Result:\n");

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  {
    SYM_TIME_SCOPE("gtsam_dynamic_d/linearize");
    for (int i = 0; i < 1000; ++i) {
      GaussianFactorGraph::shared_ptr linear_factor_graph = graph.linearize(initial);
    }
  }
}

TEST_CASE("gtsam_iterate") {
  using namespace gtsam;

  ExpressionFactorGraph graph = BuildGtsamFactors();

  // Initial values.
  Values initial;
  for (int i = 0; i < kNumPoses; ++i) {
    initial.insert(i, Pose3::identity());
  }

  // Build optimizer
  LevenbergMarquardtParams params{};
  params.setVerbosityLM("SILENT");

  LevenbergMarquardtOptimizer optimizer(graph, initial, params);
  // Values result = optimizer.optimize();
  // result.print("Final Result:\n");

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  {
    SYM_TIME_SCOPE("gtsam_dynamic_d/iterate");
    for (int i = 0; i < 1000; ++i) {
      GaussianFactorGraph::shared_ptr linear_factor_graph = optimizer.iterate();
    }
  }
}

TEST_CASE("ceres_linearize") {
  auto problem_and_vars = BuildCeresProblem();
  auto& problem = std::get<0>(problem_and_vars);

  double cost;
  std::vector<double> residuals;
  ceres::CRSMatrix jacobian;

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  {
    SYM_TIME_SCOPE("ceres_dynamic_d/linearize");
    for (int i = 0; i < 1000; ++i) {
      problem.Evaluate(ceres::Problem::EvaluateOptions{}, &cost, &residuals, nullptr, &jacobian);
    }
  }
}

TEST_CASE("ceres_iterate") {
  auto problem_and_vars = BuildCeresProblem();
  auto& problem = std::get<0>(problem_and_vars);
  auto& rotations = std::get<1>(problem_and_vars);
  auto& positions = std::get<2>(problem_and_vars);

  ceres::Solver::Options options;
  options.max_num_iterations = 200;
  options.function_tolerance = 1e-15;
  options.parameter_tolerance = 1e-30;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.logging_type = ceres::SILENT;

  ceres::Solver::Summary summary;

  // Solve - further solves should terminate in 1 iteration
  ceres::Solve(options, &problem, &summary);

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  {
    SYM_TIME_SCOPE("ceres_dynamic_d/iterate");
    for (int i = 0; i < 1000; ++i) {
      ceres::Solve(options, &problem, &summary);
    }
  }
}
