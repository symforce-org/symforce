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
#include <thread>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <gtsam/geometry/Pose3.h>
#include <sophus/se3.hpp>
#include <spdlog/spdlog.h>

#include <sym/pose3.h>
#include <symforce/opt/tic_toc.h>
#include <symforce/opt/util.h>

#include "./gen/pose_compose_point_with_jacobian0.h"
#include "./gen/pose_inverse_compose_point_with_jacobian0.h"
#include "./gen/pose_inverse_with_jacobian.h"

/**
 * Random test data utility.
 */
template <typename Scalar>
struct TestData {
  TestData(int num_poses = 1000, int num_points = 1000) {
    std::mt19937 gen(42);
    for (int i = 0; i < num_poses; ++i) {
      poses.push_back(sym::Random<sym::Pose3<Scalar>>(gen));
    }

    for (int i = 0; i < num_points; ++i) {
      points.push_back(sym::Random<sym::Vector3<Scalar>>(gen));
    }
  }

  std::vector<sym::Pose3<Scalar>> poses;
  std::vector<sym::Vector3<Scalar>> points;
};

TEMPLATE_TEST_CASE("sym_flattened", "", double, float) {
  using Scalar = TestType;

  TestData<Scalar> data;

  // Wait so perf can ignore initialization
  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  Scalar sum = 0.0;
  {
    SYM_TIME_SCOPE("sym_flattened_{}", typeid(Scalar).name());
    for (const auto& pose : data.poses) {
      for (const auto& point : data.points) {
        sym::Matrix36<Scalar> result_D_pose;
        sym::Vector3<Scalar> result =
            sym::PoseInverseComposePointWithJacobian0<Scalar>(pose, point, &result_D_pose);

        sum += result_D_pose(1, 0);
        sum += result(2);
      }
    }
  }
  spdlog::info("sym_flattened_{} sum: {}", typeid(Scalar).name(), sum);
}

TEMPLATE_TEST_CASE("sym_chained", "", double, float) {
  using Scalar = TestType;

  TestData<Scalar> data;

  // Wait so perf can ignore initialization
  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  Scalar sum = 0.0;
  {
    SYM_TIME_SCOPE("sym_chained_{}", typeid(Scalar).name());
    for (const auto& pose : data.poses) {
      for (const auto& point : data.points) {
        sym::Matrix66<Scalar> pose_inverse_D_pose;
        sym::Pose3<Scalar> pose_inverse =
            sym::PoseInverseWithJacobian<Scalar>(pose, &pose_inverse_D_pose);

        sym::Matrix36<Scalar> result_D_pose_inverse;
        sym::Vector3<Scalar> result =
            sym::PoseComposePointWithJacobian0<Scalar>(pose_inverse, point, &result_D_pose_inverse);

        sym::Matrix36<Scalar> result_D_pose = result_D_pose_inverse * pose_inverse_D_pose;

        sum += result_D_pose(1, 0);
        sum += result(2);
      }
    }
  }
  spdlog::info("sym_chained_{} sum: {}", typeid(Scalar).name(), sum);
}

TEST_CASE("gtsam_chained") {
  using Scalar = double;

  TestData<Scalar> data;
  std::vector<gtsam::Pose3> gtsam_poses;
  for (const auto& pose : data.poses) {
    gtsam_poses.push_back(gtsam::Pose3(gtsam::Rot3(pose.Rotation().Quaternion()), pose.Position()));
  }

  // Wait so perf can ignore initialization
  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  Scalar sum = 0.0;
  {
    SYM_TIME_SCOPE("gtsam_chained");
    for (const auto& pose : gtsam_poses) {
      for (const auto& point : data.points) {
        sym::Matrix66<Scalar> pose_inverse_D_pose;
        gtsam::Pose3 pose_inverse =
            gtsam::traits<gtsam::Pose3>::Inverse(pose, &pose_inverse_D_pose);

        sym::Matrix36<Scalar> result_D_pose_inverse;
        sym::Vector3<Scalar> result = pose_inverse.transformFrom(point, &result_D_pose_inverse);

        sym::Matrix36<Scalar> result_D_pose = result_D_pose_inverse * pose_inverse_D_pose;

        sum += result_D_pose(1, 0);
        sum += result(2);
      }
    }
  }

  spdlog::info("gtsam_chained sum: {}", sum);
}

TEST_CASE("gtsam_flattened") {
  using Scalar = double;

  TestData<Scalar> data;
  std::vector<gtsam::Pose3> gtsam_poses;
  for (const auto& pose : data.poses) {
    gtsam_poses.push_back(gtsam::Pose3(gtsam::Rot3(pose.Rotation().Quaternion()), pose.Position()));
  }

  // Wait so perf can ignore initialization
  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  Scalar sum = 0.0;
  {
    SYM_TIME_SCOPE("gtsam_flattened");
    for (const auto& pose : gtsam_poses) {
      for (const auto& point : data.points) {
        sym::Matrix36<Scalar> result_D_pose;
        sym::Vector3<Scalar> result = pose.transformTo(point, &result_D_pose);

        sum += result_D_pose(1, 0);
        sum += result(2);
      }
    }
  }

  spdlog::info("gtsam_flattened sum: {}", sum);
}

template <typename Scalar>
void CheckSophusJacobians(const sym::Pose3<Scalar>& pose, const sym::Vector3<Scalar>& point,
                          const sym::Matrix66<Scalar>& pose_inverse_D_pose,
                          const sym::Matrix36<Scalar>& result_D_pose_inverse,
                          const sym::Matrix36<Scalar>& result_D_pose) {
  const sym::Matrix66<Scalar> pose_inverse_D_pose_N = sym::NumericalDerivative(
      [&](const sym::Vector6<Scalar>& perturbation) -> sym::Vector6<Scalar> {
        const auto perturbed_pose = pose * Sophus::SE3<Scalar>::exp(perturbation);
        return (pose * perturbed_pose.inverse()).log();
      },
      sym::Vector6<Scalar>(sym::Vector6<Scalar>::Zero()), 0, 1e-12);
  const sym::Matrix36<Scalar> result_D_pose_inverse_N = sym::NumericalDerivative(
      [&](const sym::Vector6<Scalar>& perturbation) -> sym::Vector3<Scalar> {
        const auto perturbed_pose = pose.inverse() * Sophus::SE3<Scalar>::exp(perturbation);
        return perturbed_pose * point;
      },
      sym::Vector6<Scalar>(sym::Vector6<Scalar>::Zero()), 0, 1e-12);
  const sym::Matrix36<Scalar> result_D_pose_N = sym::NumericalDerivative(
      [&](const sym::Vector6<Scalar>& perturbation) -> sym::Vector3<Scalar> {
        const auto perturbed_pose = pose * Sophus::SE3<Scalar>::exp(perturbation);
        return perturbed_pose.inverse() * point;
      },
      sym::Vector6<Scalar>(sym::Vector6<Scalar>::Zero()), sym::kDefaultEpsilon<Scalar>,
      std::sqrt(sym::kDefaultEpsilon<Scalar>));
  spdlog::info("analytic:\n{}\n\n{}\n\n{}", pose_inverse_D_pose, result_D_pose_inverse,
               result_D_pose);
  spdlog::info("numerical:\n{}\n\n{}\n\n{}", pose_inverse_D_pose_N, result_D_pose_inverse_N,
               result_D_pose_N);
}

TEMPLATE_TEST_CASE("sophus_chained", "", double, float) {
  using Scalar = TestType;

  TestData<Scalar> data;
  std::vector<Sophus::SE3<Scalar>, Eigen::aligned_allocator<Sophus::SE3<Scalar>>> sophus_poses;
  for (const auto& pose : data.poses) {
    sophus_poses.push_back(Sophus::SE3<Scalar>(pose.Rotation().Quaternion(), pose.Position()));
  }

  // Wait so perf can ignore initialization
  std::chrono::milliseconds timespan(100);
  std::this_thread::sleep_for(timespan);

  Scalar sum = 0.0;
  {
    SYM_TIME_SCOPE("sophus_chained_{}", typeid(Scalar).name());
    for (const auto& pose : sophus_poses) {
      for (const auto& point : data.points) {
        const sym::Matrix66<Scalar> pose_inverse_D_pose = -pose.Adj();
        const Sophus::SE3<Scalar> pose_inverse = pose.inverse();

        // Sophus doesn't provide "the jacobian of composition with a point", so we have to do a
        // bunch of math by hand.  And it's still slower than symforce :)
        const sym::Vector3<Scalar> result = pose_inverse * point;

        sym::Matrix36<Scalar> result_D_pose_inverse;
        result_D_pose_inverse.template leftCols<3>() = pose_inverse.so3().matrix();
        result_D_pose_inverse.template rightCols<3>() =
            pose_inverse.so3().matrix() * Sophus::SO3<Scalar>::hat(-point);

        const sym::Matrix36<Scalar> result_D_pose = result_D_pose_inverse * pose_inverse_D_pose;

        // Checking the math we did by hand...
        // CheckSophusJacobians(pose, point, pose_inverse_D_pose, result_D_pose_inverse,
        //                      result_D_pose);

        sum += result_D_pose(1, 3 + 0);
        sum += result(2);
      }
    }
  }

  spdlog::info("sophus_chained_{} sum: {}", typeid(Scalar).name(), sum);
}
