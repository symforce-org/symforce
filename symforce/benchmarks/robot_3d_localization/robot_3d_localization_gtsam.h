/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ExpressionFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/expressions.h>

#include <symforce/examples/robot_3d_localization/common.h>
#include <symforce/examples/robot_3d_localization/gen/measurements.h>

namespace {

using namespace robot_3d_localization;

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

}  // namespace
