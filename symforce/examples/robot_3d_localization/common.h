/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once
#include <Eigen/Core>

#include <lcmtypes/sym/optimizer_params_t.hpp>

#include <sym/pose3.h>
#include <symforce/examples/robot_3d_localization/gen/measurements.h>
#include <symforce/opt/key.h>
#include <symforce/opt/optimizer.h>
#include <symforce/opt/values.h>

#include "./gen/keys.h"

namespace robot_3d_localization {

static constexpr const int kNumPoses = 5;
static constexpr const int kNumLandmarks = 20;

inline sym::optimizer_params_t RobotLocalizationOptimizerParams() {
  sym::optimizer_params_t params = sym::DefaultOptimizerParams();
  params.initial_lambda = 1e4;
  params.lambda_down_factor = 1 / 2.;
  return params;
}

template <typename Scalar>
inline sym::Values<Scalar> BuildValues(const int num_poses, const int num_landmarks) {
  sym::Values<Scalar> values;
  for (int i = 0; i < num_poses; i++) {
    values.Set(sym::Keys::WORLD_T_BODY.WithSuper(i), sym::Pose3<Scalar>());
  }

  for (int i = 0; i < num_landmarks; i++) {
    values.Set(sym::Keys::WORLD_T_LANDMARK.WithSuper(i), landmark_positions[i].cast<Scalar>());
  }

  values.Set(sym::Keys::ODOMETRY_DIAGONAL_SIGMAS,
             (sym::Vector6<Scalar>() << 0.05, 0.05, 0.05, 0.2, 0.2, 0.2).finished());

  for (int i = 0; i < num_poses - 1; i++) {
    values.Set(sym::Keys::ODOMETRY_RELATIVE_POSE_MEASUREMENTS.WithSuper(i),
               odometry_relative_pose_measurements[i].Cast<Scalar>());
  }

  values.Set(sym::Keys::MATCHING_SIGMA, Scalar(0.1));

  for (int i = 0; i < num_poses; i++) {
    for (int j = 0; j < num_landmarks; j++) {
      values.Set({sym::Keys::BODY_T_LANDMARK_MEASUREMENTS.Letter(), i, j},
                 body_t_landmark_measurements[i][j].cast<Scalar>());
    }
  }

  values.Set(sym::Keys::EPSILON, sym::kDefaultEpsilon<Scalar>);

  return values;
}

}  // namespace robot_3d_localization
