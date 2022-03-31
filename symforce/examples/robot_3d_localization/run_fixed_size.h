/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <symforce/opt/factor.h>

namespace robot_3d_localization {

void RunFixed();

template <typename Scalar>
sym::Factor<Scalar> BuildFixedFactor();

extern template sym::Factor<double> BuildFixedFactor<double>();
extern template sym::Factor<float> BuildFixedFactor<float>();

}  // namespace robot_3d_localization
