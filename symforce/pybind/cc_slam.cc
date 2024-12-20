/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./cc_slam.h"

#include <pybind11/eigen.h>

#include <symforce/slam/imu_preintegration/imu_preintegrator.h>
#include <symforce/slam/imu_preintegration/preintegrated_imu_measurements.h>

#include "./lcm_type_casters.h"
#include "./sym_type_casters.h"

namespace py = pybind11;

namespace sym {

void AddSlamWrapper(pybind11::module_ module) {
  auto pim = py::class_<sym::PreintegratedImuMeasurementsd>(
      module, "PreintegratedImuMeasurements",
      "Struct of Preintegrated IMU Measurements (not including the covariance of change in "
      "orientation, velocity, and position).");

  py::class_<sym::PreintegratedImuMeasurementsd::Delta>(
      pim, "Delta", "A convenient struct that holds the Preintegrated delta.")
      .def_static("from_lcm", &sym::PreintegratedImuMeasurementsd::Delta::FromLcm)
      .def("get_lcm_type", &sym::PreintegratedImuMeasurementsd::Delta::GetLcmType)
      .def("roll_forward_state", &sym::PreintegratedImuMeasurementsd::Delta::RollForwardState,
           py::arg("pose_i"), py::arg("vel_i"), py::arg("gravity"))
      .def_readwrite("Dt", &sym::PreintegratedImuMeasurementsd::Delta::Dt)
      .def_readwrite("DR", &sym::PreintegratedImuMeasurementsd::Delta::DR)
      .def_readwrite("Dv", &sym::PreintegratedImuMeasurementsd::Delta::Dv)
      .def_readwrite("Dp", &sym::PreintegratedImuMeasurementsd::Delta::Dp);

  pim.def_static("from_lcm", &sym::PreintegratedImuMeasurementsd::FromLcm)
      .def(py::init<const sym::Vector3d&, const sym::Vector3d&>(), py::arg("accel_bias"),
           py::arg("gyro_bias"))
      .def("get_bias_corrected_delta", &sym::PreintegratedImuMeasurementsd::GetBiasCorrectedDelta,
           py::arg("new_accel_bias"), py::arg("new_gyro_bias"))
      .def("get_lcm_type", &sym::PreintegratedImuMeasurementsd::GetLcmType)
      .def_readwrite("accel_bias", &sym::PreintegratedImuMeasurementsd::accel_bias)
      .def_readwrite("gyro_bias", &sym::PreintegratedImuMeasurementsd::gyro_bias)
      .def_readwrite("delta", &sym::PreintegratedImuMeasurementsd::delta)
      .def_readwrite("DR_D_gyro_bias", &sym::PreintegratedImuMeasurementsd::DR_D_gyro_bias)
      .def_readwrite("Dv_D_accel_bias", &sym::PreintegratedImuMeasurementsd::Dv_D_accel_bias)
      .def_readwrite("Dv_D_gyro_bias", &sym::PreintegratedImuMeasurementsd::Dv_D_gyro_bias)
      .def_readwrite("Dp_D_accel_bias", &sym::PreintegratedImuMeasurementsd::Dp_D_accel_bias)
      .def_readwrite("Dp_D_gyro_bias", &sym::PreintegratedImuMeasurementsd::Dp_D_gyro_bias);

  py::class_<sym::ImuPreintegratord>(
      module, "ImuPreintegrator",
      "Class to on-manifold preintegrate IMU measurements for usage in a SymForce optimization "
      "problem.")
      .def(py::init<const sym::Vector3d&, const sym::Vector3d&>(), py::arg("accel_bias"),
           py::arg("gyro_bias"))
      .def("integrate_measurement", &sym::ImuPreintegratord::IntegrateMeasurement,
           py::arg("measured_accel"), py::arg("measured_gyro"), py::arg("accel_cov"),
           py::arg("gyro_cov"), py::arg("dt"), py::arg("epsilon") = kDefaultEpsilond)
      .def("preintegrated_measurements", &sym::ImuPreintegratord::PreintegratedMeasurements)
      .def("covariance", &sym::ImuPreintegratord::Covariance);
}

}  // namespace sym
