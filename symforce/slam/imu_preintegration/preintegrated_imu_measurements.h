/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Core>

#include <lcmtypes/sym/imu_integrated_measurement_t.hpp>

#include <sym/pose3.h>
#include <sym/rot3.h>
#include <sym/util/typedefs.h>

namespace sym {

/**
 * Struct of Preintegrated IMU Measurements (not including the covariance of change in
 * orientation, velocity, and position).
 */
template <typename ScalarType>
struct PreintegratedImuMeasurements {
  using Scalar = ScalarType;
  using Rot3 = sym::Rot3<Scalar>;
  using Vector3 = sym::Vector3<Scalar>;
  using Matrix33 = sym::Matrix33<Scalar>;

  /// A convenient struct that holds the Preintegrated delta
  struct Delta {
    // Constructor from LCM type
    static Delta FromLcm(const imu_integrated_measurement_delta_t& msg);

    // Converts this to a LCM type
    imu_integrated_measurement_delta_t GetLcmType() const;

    // Rolls forward the given state by this Delta
    std::pair<Pose3<Scalar>, Vector3> RollForwardState(const Pose3<Scalar>& pose_i,
                                                       const Vector3& vel_i,
                                                       const Vector3& gravity) const;

    // The elapsed time of the measurement period
    Scalar Dt{0};

    // The rotation that occurred over the measurement period; i.e., maps the coordinates of a
    // vector in the body frame of the end of the measurement period to the coordinates of the
    // vector in the body frame at the start of the measurement period
    Rot3 DR{Rot3::Identity()};

    // The velocity change that occurred over the measurement period in the body frame of the
    // initial measurement (assuming 0 acceleration due to gravity)
    Vector3 Dv{Vector3::Zero()};

    // The position change that occurred over the measurement period in the body frame of the
    // initial measurement (assuming 0 acceleration due to gravity and 0 initial velocity)
    Vector3 Dp{Vector3::Zero()};
  };

  // Constructor from LCM type.
  static PreintegratedImuMeasurements<Scalar> FromLcm(const imu_integrated_measurement_t& msg);

  /// Initialize instance struct with accel_bias and gyro_bias and all other values
  /// zeroed out (scalars, vectors, and matrices) or set to the identity (DR).
  PreintegratedImuMeasurements(const Vector3& accel_bias, const Vector3& gyro_bias);

  // Given new accel and gyro biases, return a first-order correction to the preintegrated delta
  // The user is responsible for making sure that the new biases are sufficiently close to the
  // original biases used during the preintegration.
  Delta GetBiasCorrectedDelta(const Vector3& new_accel_bias, const Vector3& new_gyro_bias) const;

  // Converts this to a LCM type
  imu_integrated_measurement_t GetLcmType() const;

  // --------------------------------------------------------------------------
  // StorageOps concept
  // --------------------------------------------------------------------------

  static constexpr int32_t StorageDim() {
    return StorageOps<PreintegratedImuMeasurements>::StorageDim();
  }

  void ToStorage(Scalar* const vec) const;
  static PreintegratedImuMeasurements FromStorage(const Scalar* vec);

  // The original accelerometer bias used during preintegration
  Vector3 accel_bias;

  // The original gyroscope bias used during preintegration
  Vector3 gyro_bias;

  /// See description for Delta.  This is the Delta for the biases used during preintegration, i.e.
  /// for PreintegratedImuMeasurements::accel_bias and PreintegratedImuMeasurements::gyro_bias.  For
  /// the (approximate) Delta with other biases, see GetBiasCorrectedDelta
  Delta delta;

  // Derivatives of DR/Dv/Dp w.r.t. the gyroscope/accelerometer bias linearized at the values
  // of gyro_bias and accel_bias
  Matrix33 DR_D_gyro_bias;
  Matrix33 Dv_D_accel_bias;
  Matrix33 Dv_D_gyro_bias;
  Matrix33 Dp_D_accel_bias;
  Matrix33 Dp_D_gyro_bias;
};

template <typename ScalarType>
struct StorageOps<PreintegratedImuMeasurements<ScalarType>> {
  using T = PreintegratedImuMeasurements<ScalarType>;
  using Scalar = ScalarType;

  static constexpr int32_t StorageDim() {
    const auto delta_storage_dim = 1 + 4 + 3 + 3;
    return 3 + 3 + delta_storage_dim + 3 * 3 * 5;
  }

  static void ToStorage(const T& a, ScalarType* out);
  static T FromStorage(const ScalarType* data);

  static constexpr type_t TypeEnum() {
    return type_t::PREINTEGRATED_IMU_MEASUREMENTS;
  }
};

using PreintegratedImuMeasurementsd = PreintegratedImuMeasurements<double>;
using PreintegratedImuMeasurementsf = PreintegratedImuMeasurements<float>;

static_assert(sizeof(PreintegratedImuMeasurementsd) ==
              StorageOps<PreintegratedImuMeasurementsd>::StorageDim() * sizeof(double));
static_assert(alignof(PreintegratedImuMeasurementsd) == sizeof(double));
static_assert(sizeof(PreintegratedImuMeasurementsf) ==
              StorageOps<PreintegratedImuMeasurementsf>::StorageDim() * sizeof(float));
static_assert(alignof(PreintegratedImuMeasurementsf) == sizeof(float));

}  // namespace sym

// Explicit instantiation
extern template struct sym::PreintegratedImuMeasurements<double>;
extern template struct sym::PreintegratedImuMeasurements<float>;
extern template struct sym::StorageOps<sym::PreintegratedImuMeasurements<double>>;
extern template struct sym::StorageOps<sym::PreintegratedImuMeasurements<float>>;
