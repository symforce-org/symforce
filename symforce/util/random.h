#pragma once

#include <Eigen/Dense>
#include <geo/pose2.h>
#include <geo/pose3.h>
#include <geo/rot2.h>
#include <geo/rot3.h>

namespace sym {

template <typename Scalar, size_t V>
Eigen::Matrix<Scalar, V, 1> RandomNormalVector(std::mt19937& gen) {
  std::normal_distribution<Scalar> distribution{};
  const Eigen::Matrix<Scalar, V, 1> translation =
      Eigen::Matrix<Scalar, V, 1>::NullaryExpr([&]() { return distribution(gen); });
  return translation;
}

template <typename T>
T Random(std::mt19937& gen);

// Specializations for doubles

template <>
geo::Rot2d Random<geo::Rot2d>(std::mt19937& gen) {
  return geo::Rot2d::Random(gen);
}

template <>
geo::Rot3d Random<geo::Rot3d>(std::mt19937& gen) {
  return geo::Rot3d::Random(gen);
}

template <>
geo::Pose2d Random<geo::Pose2d>(std::mt19937& gen) {
  const geo::Rot2d rotation = Random<geo::Rot2d>(gen);
  const Eigen::Matrix<double, 2, 1> translation = RandomNormalVector<double, 2>(gen);
  return geo::Pose2d(rotation, translation);
}

template <>
geo::Pose3d Random<geo::Pose3d>(std::mt19937& gen) {
  const geo::Rot3d rotation = Random<geo::Rot3d>(gen);
  const Eigen::Matrix<double, 3, 1> translation = RandomNormalVector<double, 3>(gen);
  return geo::Pose3d(rotation, translation);
}

// Specializations for floats

template <>
geo::Rot2f Random<geo::Rot2f>(std::mt19937& gen) {
  return geo::Rot2f::Random(gen);
}

template <>
geo::Rot3f Random<geo::Rot3f>(std::mt19937& gen) {
  return geo::Rot3f::Random(gen);
}

template <>
geo::Pose2f Random<geo::Pose2f>(std::mt19937& gen) {
  const geo::Rot2f rotation = Random<geo::Rot2f>(gen);
  const Eigen::Matrix<float, 2, 1> translation = RandomNormalVector<float, 2>(gen);
  return geo::Pose2f(rotation, translation);
}

template <>
geo::Pose3f Random<geo::Pose3f>(std::mt19937& gen) {
  const geo::Rot3f rotation = Random<geo::Rot3f>(gen);
  const Eigen::Matrix<float, 3, 1> translation = RandomNormalVector<float, 3>(gen);
  return geo::Pose3f(rotation, translation);
}

}  // namespace sym
