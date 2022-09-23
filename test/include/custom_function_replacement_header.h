/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#pragma once

#include <cmath>

namespace fast_math {

template <typename Scalar>
Scalar sin(const Scalar x) {
  return std::sin(x) + 1.0;  // Test function to ensure a different result from std::sin
}

}  // namespace fast_math
