/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <vector>

#include <symforce/opt/factor.h>

namespace robot_3d_localization {

void RunDynamic();

/**
 * Creates a factor for a prior on the relative pose between view i and view j
 */
template <typename Scalar>
sym::Factor<Scalar> CreateMatchingFactor(int i, int j);

template <typename Scalar>
sym::Factor<Scalar> CreateOdometryFactor(int i);

template <typename Scalar>
std::vector<sym::Factor<Scalar>> BuildDynamicFactors(int num_poses, int num_landmarks);

extern template sym::Factor<double> CreateMatchingFactor<double>(int i, int j);
extern template sym::Factor<float> CreateMatchingFactor<float>(int i, int j);

extern template sym::Factor<double> CreateOdometryFactor<double>(int i);
extern template sym::Factor<float> CreateOdometryFactor<float>(int i);

extern template std::vector<sym::Factor<double>> BuildDynamicFactors<double>(int num_poses,
                                                                             int num_landmarks);
extern template std::vector<sym::Factor<float>> BuildDynamicFactors<float>(int num_poses,
                                                                           int num_landmarks);

}  // namespace robot_3d_localization
