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
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/expressions.h>
#include <spdlog/spdlog.h>

#include <sym/rot3.h>
#include <symforce/examples/robot_3d_localization/common.h>
#include <symforce/examples/robot_3d_localization/gen/cpp/symforce/robot_3d_localization/linearization.h>
#include <symforce/examples/robot_3d_localization/gen/cpp/symforce/sym/keys.h>
#include <symforce/examples/robot_3d_localization/gen/cpp/symforce/sym/matching_factor.h>
#include <symforce/examples/robot_3d_localization/gen/cpp/symforce/sym/odometry_factor.h>
#include <symforce/examples/robot_3d_localization/gen/measurements.h>
#include <symforce/examples/robot_3d_localization/run_dynamic_size.h>
#include <symforce/examples/robot_3d_localization/run_fixed_size.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>
#include <symforce/opt/tic_toc.h>

#include "catch.hpp"

using namespace robot_3d_localization;

// ----------------------------------------------------------------------------
// GTSAM
// ----------------------------------------------------------------------------

gtsam::ExpressionFactorGraph BuildGtsamFactors() {
  gtsam::ExpressionFactorGraph graph;

  // Initial expressions.
  std::vector<gtsam::Expression<gtsam::Pose3>> world_T_body_vec;
  for (int i = 0; i < kNumPoses; ++i) {
    world_T_body_vec.emplace_back(i);
  }
  std::vector<gtsam::Expression<gtsam::Point3>> world_t_landmark_vec;
  for (int i = 0; i < kNumLandmarks; ++i) {
    world_t_landmark_vec.emplace_back(landmark_positions[i]);
  }

  // Add matching residuals.
  auto matching_sigma = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  for (int pose_idx = 0; pose_idx < kNumPoses; ++pose_idx) {
    for (int landmark_idx = 0; landmark_idx < kNumLandmarks; ++landmark_idx) {
      graph.addExpressionFactor(
          gtsam::Expression<gtsam::Point3>(&gtsam::Pose3::transformTo, world_T_body_vec[pose_idx],
                                           world_t_landmark_vec[landmark_idx]),
          body_t_landmark_measurements[pose_idx][landmark_idx], matching_sigma);
    }
  }

  // Add odometry residuals.
  auto odometry_noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.2, 0.2, 0.2).finished());
  for (int pose_idx = 0; pose_idx < kNumPoses - 1; ++pose_idx) {
    const sym::Pose3d sym_pose = odometry_relative_pose_measurements[pose_idx];
    const gtsam::Pose3 pose(gtsam::Rot3(sym_pose.Rotation().Quaternion()), sym_pose.Position());
    graph.addExpressionFactor(
        gtsam::between(world_T_body_vec[pose_idx], world_T_body_vec[pose_idx + 1]), pose,
        odometry_noise);
  }

  return graph;
}

// ----------------------------------------------------------------------------
// Ceres
// ----------------------------------------------------------------------------

// this is from `examples/slam/pose_graph_3d` example in ceres-solver
class RelativePoseError {
 public:
  RelativePoseError(const Eigen::Quaterniond& a_q_b, const Eigen::Vector3d& a_t_b,
                    const Eigen::Matrix<double, 6, 6>& info_matrix)
      : a_q_b_measured_(a_q_b), a_t_b_measured_(a_t_b), info_matrix_(info_matrix) {}

  template <typename T>
  bool operator()(const T* const world_q_a_ptr, const T* const world_t_a_ptr,
                  const T* const world_q_b_ptr, const T* const world_t_b_ptr,
                  T* residuals_ptr) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> world_t_a(world_t_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> world_q_a(world_q_a_ptr);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> world_t_b(world_t_b_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> world_q_b(world_q_b_ptr);

    Eigen::Quaternion<T> a_q_b_estimated = world_q_a.conjugate() * world_q_b;

    Eigen::Matrix<T, 3, 1> a_t_b_estimated = world_q_a.conjugate() * (world_t_b - world_t_a);

    Eigen::Quaternion<T> a_estimated_q_a_measured =
        a_q_b_estimated.conjugate() * a_q_b_measured_.template cast<T>();

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = T(2.0) * a_estimated_q_a_measured.vec();
    residuals.template block<3, 1>(3, 0) = a_t_b_estimated - a_t_b_measured_.template cast<T>();

    residuals.applyOnTheLeft(info_matrix_.template cast<T>());

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Quaterniond& a_q_b, const Eigen::Vector3d& a_t_b,
                                     const Eigen::Matrix<double, 6, 6>& info_matrix) {
    return new ceres::AutoDiffCostFunction<RelativePoseError, 6, 4, 3, 4, 3>(
        new RelativePoseError(a_q_b, a_t_b, info_matrix));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  const Eigen::Quaterniond a_q_b_measured_;
  const Eigen::Vector3d a_t_b_measured_;
  const Eigen::Matrix<double, 6, 6> info_matrix_;
};

class ScanMatchingError {
 public:
  ScanMatchingError(const Eigen::Vector3d& body_t_landmark, const Eigen::Vector3d& world_t_landmark,
                    const double sigma)
      : body_t_landmark_(body_t_landmark), world_t_landmark_(world_t_landmark), sigma_(sigma) {}

  template <typename T>
  bool operator()(const T* const world_q_body_ptr, const T* const world_t_body_ptr,
                  T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> world_t_body(world_t_body_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> world_q_body(world_q_body_ptr);

    Eigen::Matrix<T, 3, 3> body_R_world = world_q_body.conjugate().toRotationMatrix();

    Eigen::Matrix<T, 3, 1> body_t_landmark_predict =
        body_R_world * world_t_landmark_.template cast<T>() - body_R_world * world_t_body;

    Eigen::Matrix<T, 3, 1> diff = body_t_landmark_predict - body_t_landmark_.template cast<T>();

    residuals[0] = (diff[0]) / T(sigma_);
    residuals[1] = (diff[1]) / T(sigma_);
    residuals[2] = (diff[2]) / T(sigma_);

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& body_t_landmark,
                                     const Eigen::Vector3d& world_t_landmark, const double sigma) {
    return new ceres::AutoDiffCostFunction<ScanMatchingError, 3, 4, 3>(
        new ScanMatchingError(body_t_landmark, world_t_landmark, sigma));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  const Eigen::Vector3d body_t_landmark_;
  const Eigen::Vector3d world_t_landmark_;
  const double sigma_;
};

auto BuildCeresProblem() {
  // initialize poses
  std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> rotations;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> positions;
  Eigen::Quaterniond q(0, 0, 0, 1);
  Eigen::Vector3d t(0, 0, 0);
  for (int i = 0; i < kNumPoses; ++i) {
    rotations.push_back(q);
    positions.push_back(t);
  }

  // Set up a problem
  ceres::Problem problem;

  ceres::LocalParameterization* quaternion_local_parameterization =
      new ceres::EigenQuaternionParameterization;

  // Set up odom residuals
  Eigen::Matrix<double, 6, 1> odometry_info_diagonal;
  odometry_info_diagonal << 1 / 0.05, 1 / 0.05, 1 / 0.05, 1 / 0.2, 1 / 0.2, 1 / 0.2;
  for (int i = 0; i < odometry_relative_pose_measurements.size(); ++i) {
    Eigen::Quaterniond a_q_b = odometry_relative_pose_measurements[i].Rotation().Quaternion();
    Eigen::Vector3d a_t_b = odometry_relative_pose_measurements[i].Position();

    ceres::CostFunction* cost_function =
        RelativePoseError::Create(a_q_b, a_t_b, odometry_info_diagonal.asDiagonal());
    problem.AddResidualBlock(cost_function, nullptr, rotations[i].coeffs().data(),
                             positions[i].data(), rotations[i + 1].coeffs().data(),
                             positions[i + 1].data());

    problem.SetParameterization(rotations[i].coeffs().data(), quaternion_local_parameterization);
    problem.SetParameterization(rotations[i + 1].coeffs().data(),
                                quaternion_local_parameterization);
  }

  // Set up landmark residuals
  double landmark_sigma = 0.1;
  for (int pose_idx = 0; pose_idx < body_t_landmark_measurements.size(); ++pose_idx) {
    for (int lm_idx = 0; lm_idx < body_t_landmark_measurements[pose_idx].size(); ++lm_idx) {
      Eigen::Vector3d world_t_landmark = landmark_positions[lm_idx];

      ceres::CostFunction* cost_function = ScanMatchingError::Create(
          body_t_landmark_measurements[pose_idx][lm_idx], world_t_landmark, landmark_sigma);
      problem.AddResidualBlock(cost_function, nullptr, rotations[pose_idx].coeffs().data(),
                               positions[pose_idx].data());
    }
  }

  // Eek, this only works because move(vector) is guaranteed to not move the backing array
  return std::make_tuple(std::move(problem), std::move(rotations), std::move(positions));
}

// ----------------------------------------------------------------------------
// Test Cases
// ----------------------------------------------------------------------------

TEMPLATE_TEST_CASE("sym_dynamic_linearize", "", double, float) {
  using Scalar = TestType;

  sym::Values<Scalar> values = BuildValues<Scalar>(kNumPoses, kNumLandmarks);

  // Create and set up Optimizer
  const std::vector<sym::Factor<Scalar>> factors =
      BuildDynamicFactors<Scalar>(kNumPoses, kNumLandmarks);

  sym::Optimizer<Scalar> optimizer(RobotLocalizationOptimizerParams(), factors,
                                   sym::kDefaultEpsilon<Scalar>, "sym_dynamic_linearize");

  sym::Linearizer<Scalar>& linearizer = optimizer.Linearizer();
  sym::Linearization<Scalar> linearization;

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  // Linearize
  {
    SYM_TIME_SCOPE("sym_dynamic_{}/linearize", typeid(Scalar).name());
    for (int i = 0; i < 1000; i++) {
      linearizer.Relinearize(values, &linearization);
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

  sym::OptimizationStats<Scalar> stats;

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  // Iterate
  {
    SYM_TIME_SCOPE("sym_dynamic_{}/iterate", typeid(Scalar).name());
    for (int i = 0; i < 1000; i++) {
      optimizer.Optimize(&values, /* num_iterations */ 1, /* populate_best_linearization */ false,
                         &stats);
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
  sym::Linearization<Scalar> linearization;

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  // Linearize
  {
    SYM_TIME_SCOPE("sym_fixed_{}/linearize", typeid(Scalar).name());
    for (int i = 0; i < 1000; i++) {
      linearizer.Relinearize(values, &linearization);
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

  sym::OptimizationStats<Scalar> stats;

  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  // Iterate
  {
    SYM_TIME_SCOPE("sym_fixed_{}/iterate", typeid(Scalar).name());
    for (int i = 0; i < 1000; i++) {
      optimizer.Optimize(&values, /* num_iterations */ 1, /* populate_best_linearization */ false,
                         &stats);
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
