#pragma once

#include <vector>

#include <symforce/opt/factor.h>

namespace robot_3d_localization {

void RunDynamic();

/**
 * Creates a factor for a prior on the relative pose between view i and view j
 */
template <typename Scalar>
sym::Factor<Scalar> CreateMatchingFactor(const int i, const int j);

template <typename Scalar>
sym::Factor<Scalar> CreateOdometryFactor(const int i);

template <typename Scalar>
std::vector<sym::Factor<Scalar>> BuildDynamicFactors(const int num_poses, const int num_landmarks);

extern template sym::Factor<double> CreateMatchingFactor<double>(const int i, const int j);
extern template sym::Factor<float> CreateMatchingFactor<float>(const int i, const int j);

extern template sym::Factor<double> CreateOdometryFactor<double>(const int i);
extern template sym::Factor<float> CreateOdometryFactor<float>(const int i);

extern template std::vector<sym::Factor<double>> BuildDynamicFactors<double>(
    const int num_poses, const int num_landmarks);
extern template std::vector<sym::Factor<float>> BuildDynamicFactors<float>(const int num_poses,
                                                                           const int num_landmarks);

}  // namespace robot_3d_localization
