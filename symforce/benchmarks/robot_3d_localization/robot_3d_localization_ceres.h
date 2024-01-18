/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Core>
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <symforce/examples/robot_3d_localization/common.h>
#include <symforce/examples/robot_3d_localization/gen/measurements.h>

namespace {

using namespace robot_3d_localization;

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

}  // namespace
