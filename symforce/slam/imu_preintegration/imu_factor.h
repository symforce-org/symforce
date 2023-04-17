/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <vector>

#include <Eigen/Core>

#include <sym/pose3.h>
#include <symforce/opt/factor.h>
#include <symforce/opt/key.h>

#include "./imu_preintegrator.h"
#include "./preintegrated_imu_measurements.h"

namespace sym {

/**
 * A factor for using on-manifold IMU preintegration in a SymForce optimization problem. By
 * on-manifold, it is meant that the angular velocity measurements are composed as rotations
 * rather than tangent-space approximations.
 *
 * Assumes IMU bias is constant over the preintegration window. Does not recompute the
 * preintegrated measurements when the IMU bias estimate changes during optimization, but rather
 * uses a first order approximation linearized at the IMU biases given during preintegration.
 *
 * The gravity argument should be [0, 0, -g] (where g is 9.8, assuming your world frame is
 * gravity aligned so that the -z direction points towards the Earth) unless your IMU reads 0
 * acceleration while stationary, in which case it should be [0, 0, 0].
 *
 * More generally, the gravity argument should yield the true acceleration when added to the
 * accelerometer measurement (after the measurement has been converted to the world frame).
 *
 * Is a functor so as to store the preintegrated IMU measurements between two times.
 * The residual computed is the local-coordinate difference between the final state (pose
 * and velocity) and the state you would expect given the initial state and the IMU measurements.
 *
 * Based heavily on the on manifold ImuFactor found in GTSAM and on the paper:
 *
 * Christian Forster, Luca Carlone, Frank Dellaert, and Davide Scaramuzza,
 * “IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation”,
 * Robotics: Science and Systems (RSS), 2015.
 *
 * Example Usage:
 *
 *   enum Var : char {
 *     POSE = 'p',        // Pose3d
 *     VELOCITY = 'v',    // Vector3d
 *     ACCEL_BIAS = 'A',  // Vector3d
 *     GYRO_BIAS = 'G',   // Vector3d
 *     GRAVITY = 'g',     // Vector3d
 *     EPSILON = 'e'      // Scalar
 *   };
 *
 *   struct ImuMeasurement {
 *     Eigen::Vector3d acceleration;
 *     Eigen::Vector3d angular_velocity;
 *     double duration;
 *   };
 *
 *   int main() {
 *     // Dummy function declarations
 *     std::vector<ImuMeasurement> GetMeasurementsBetween(double start_time, double end_time);
 *     std::vector<double> GetKeyFrameTimes();
 *     Eigen::Vector3d GetAccelBiasEstimate(double time);
 *     Eigen::Vector3d GetGyroBiasEstimate(double time);
 *     void AppendOtherFactors(std::vector<sym::Factord> & factors);
 *
 *     // Example accelerometer and gyroscope covariances
 *     const Eigen::Vector3d accel_cov = Eigen::Vector3d::Constant(1e-5);
 *     const Eigen::Vector3d gyro_cov = Eigen::Vector3d::Constant(1e-5);
 *
 *     std::vector<sym::Factord> factors;
 *
 *     // GetKeyFrameTimes assumed to return at least 2 times
 *     std::vector<double> key_frame_times = GetKeyFrameTimes();
 *
 *     // Integrating Imu measurements between keyframes, creating an ImuFactor for each
 *     for (int i = 0; i < static_cast<int>(key_frame_times.size()) - 1; i++) {
 *       const double start_time = key_frame_times[i];
 *       const double end_time = key_frame_times[i + 1];
 *
 *       const std::vector<ImuMeasurement> measurements = GetMeasurementsBetween(start_time,
 *                                                                               end_time);
 *
 *       // ImuPreintegrator should be initialized with the best guesses for the IMU biases
 *       sym::ImuPreintegrator<double> integrator(GetAccelBiasEstimate(start_time),
 *                                                GetGyroBiasEstimate(start_time));
 *       for (const ImuMeasurement& meas : measurements) {
 *         integrator.IntegrateMeasurement(meas.acceleration, meas.angular_velocity, accel_cov,
 *                                         gyro_cov, meas.duration);
 *       }
 *
 *       factors.push_back(sym::ImuFactor<double>(integrator)
 *                             .Factor({{Var::POSE, i},
 *                                      {Var::VELOCITY, i},
 *                                      {Var::POSE, i + 1},
 *                                      {Var::VELOCITY, i + 1},
 *                                      {Var::ACCEL_BIAS, i},
 *                                      {Var::GYRO_BIAS, i},
 *                                      {Var::GRAVITY},
 *                                      {Var::EPSILON}}));
 *     }
 *
 *     // Adding any other factors there might be for the problem
 *     AppendOtherFactors(factors);
 *
 *     sym::Optimizerd optimizer(sym::DefaultOptimizerParams(), factors);
 *
 *     // Build Values
 *     sym::Valuesd values;
 *     for (int i = 0; i < key_frame_times.size(); i++) {
 *       values.Set({Var::POSE, i}, sym::Pose3d());
 *       values.Set({Var::VELOCITY, i}, Eigen::Vector3d::Zero());
 *     }
 *     for (int i = 0; i < key_frame_times.size() - 1; i++) {
 *       values.Set({Var::ACCEL_BIAS, i}, GetAccelBiasEstimate(key_frame_times[i]));
 *       values.Set({Var::GYRO_BIAS, i}, GetGyroBiasEstimate(key_frame_times[i]));
 *     }
 *     // gravity should point towards the direction of acceleration
 *     values.Set({Var::GRAVITY}, Eigen::Vector3d(0.0, 0.0, -9.8));
 *     values.Set({Var::EPSILON}, sym::kDefaultEpsilond);
 *     // Initialize any other items in values ...
 *
 *     optimizer.Optimize(values);
 *   }
 */
template <typename Scalar>
class ImuFactor {
 private:
  PreintegratedImuMeasurements<Scalar> preintegrated_measurements_;
  Eigen::Matrix<Scalar, 9, 9> sqrt_info_;

 public:
  /**
   * Construct an ImuFactor connecting two states from the (preintegrated) imu measurements
   * between them.
   */
  ImuFactor(const ImuPreintegrator<Scalar>& preintegrator);

  /**
   * Construct a Factor object that can be passed to an Optimizer object given the keys to
   * be passed to the function.
   *
   * Convenience method to avoid manually specifying which arguments are optimized.
   */
  sym::Factor<Scalar> Factor(const std::vector<Key>& keys_to_func) const;

  /**
   * Calculate a between factor on the product manifold of the pose and velocity where the prior
   * is calculated from the preintegrated IMU measurements.
   *
   * Time step i is the time of the first IMU measurement of the interval.
   * Time step j is the time after the last IMU measurement of the interval.
   *
   * Args:
   *     pose_i: Pose at time step i (world_T_body)
   *     vel_i: Velocity at time step i (world frame)
   *     pose_j: Pose at time step j (world_T_body)
   *     vel_j: Velocity at time step j (world frame)
   *     accel_bias_i: The bias of the accelerometer measurements between timesteps i and j
   *     gyro_bias_i: The bias of the gyroscope measurements between timesteps i and j
   *     gravity: Acceleration due to gravity (in the same frame as pose_x and vel_x),
   *              i.e., the vector which when added to the accelerometer measurements
   *              gives the true acceleration (up to bias and noise) of the IMU.
   *     epsilon: epsilon used for numerical stability
   *
   * Outputs:
   *     res: The 9dof whitened local coordinate difference between predicted and estimated state
   *     jacobian: (9x24) jacobian of res wrt args pose_i (6), vel_i (3), pose_j (6), vel_j (3),
   *               accel_bias_i (3), gyro_bias_i (3)
   *     hessian: (24x24) Gauss-Newton hessian for args pose_i (6), vel_i (3), pose_j (6),
   *              vel_j (3), accel_bias_i (3), gyro_bias_i (3)
   *     rhs: (24x1) Gauss-Newton rhs for args pose_i (6), vel_i (3), pose_j (6), vel_j (3),
   *          accel_bias_i (3), gyro_bias_i (3)
   */
  void operator()(const sym::Pose3<Scalar>& pose_i, const Eigen::Matrix<Scalar, 3, 1>& vel_i,
                  const sym::Pose3<Scalar>& pose_j, const Eigen::Matrix<Scalar, 3, 1>& vel_j,
                  const Eigen::Matrix<Scalar, 3, 1>& accel_bias_i,
                  const Eigen::Matrix<Scalar, 3, 1>& gyro_bias_i,
                  const Eigen::Matrix<Scalar, 3, 1>& gravity, const Scalar epsilon,
                  Eigen::Matrix<Scalar, 9, 1>* const res = nullptr,
                  Eigen::Matrix<Scalar, 9, 24>* const jacobian = nullptr,
                  Eigen::Matrix<Scalar, 24, 24>* const hessian = nullptr,
                  Eigen::Matrix<Scalar, 24, 1>* const rhs = nullptr) const;
};

using ImuFactord = ImuFactor<double>;
using ImuFactorf = ImuFactor<float>;

}  // namespace sym

extern template class sym::ImuFactor<double>;
extern template class sym::ImuFactor<float>;
