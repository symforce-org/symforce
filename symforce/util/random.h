#pragma once

#include <Eigen/Dense>
#include <geo/pose2.h>
#include <geo/pose3.h>
#include <geo/rot2.h>
#include <geo/rot3.h>

namespace sym {

template <typename Scalar, size_t M, size_t N>
Eigen::Matrix<Scalar, M, N> RandomNormalMatrix(std::mt19937& gen) {
  std::normal_distribution<Scalar> distribution{};
  const Eigen::Matrix<Scalar, M, N> matrix =
      Eigen::Matrix<Scalar, M, N>::NullaryExpr([&]() { return distribution(gen); });
  return matrix;
}

template <typename Scalar, size_t V>
Eigen::Matrix<Scalar, V, 1> RandomNormalVector(std::mt19937& gen) {
  return RandomNormalMatrix<Scalar, V, 1>(gen);
}

template <typename T>
T Random(std::mt19937& gen);

// Specializations for doubles

template <>
sym::Rot2d Random<sym::Rot2d>(std::mt19937& gen) {
  return sym::Rot2d::Random(gen);
}

template <>
sym::Rot3d Random<sym::Rot3d>(std::mt19937& gen) {
  return sym::Rot3d::Random(gen);
}

template <>
sym::Pose2d Random<sym::Pose2d>(std::mt19937& gen) {
  const sym::Rot2d rotation = Random<sym::Rot2d>(gen);
  const Eigen::Matrix<double, 2, 1> translation = RandomNormalVector<double, 2>(gen);
  return sym::Pose2d(rotation, translation);
}

template <>
sym::Pose3d Random<sym::Pose3d>(std::mt19937& gen) {
  const sym::Rot3d rotation = Random<sym::Rot3d>(gen);
  const Eigen::Matrix<double, 3, 1> translation = RandomNormalVector<double, 3>(gen);
  return sym::Pose3d(rotation, translation);
}

// Specializations for floats

template <>
sym::Rot2f Random<sym::Rot2f>(std::mt19937& gen) {
  return sym::Rot2f::Random(gen);
}

template <>
sym::Rot3f Random<sym::Rot3f>(std::mt19937& gen) {
  return sym::Rot3f::Random(gen);
}

template <>
sym::Pose2f Random<sym::Pose2f>(std::mt19937& gen) {
  const sym::Rot2f rotation = Random<sym::Rot2f>(gen);
  const Eigen::Matrix<float, 2, 1> translation = RandomNormalVector<float, 2>(gen);
  return sym::Pose2f(rotation, translation);
}

template <>
sym::Pose3f Random<sym::Pose3f>(std::mt19937& gen) {
  const sym::Rot3f rotation = Random<sym::Rot3f>(gen);
  const Eigen::Matrix<float, 3, 1> translation = RandomNormalVector<float, 3>(gen);
  return sym::Pose3f(rotation, translation);
}

}  // namespace sym
