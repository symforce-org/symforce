/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./linearization.h"

// Explicit instantiation
template struct sym::Linearization<double>;
template struct sym::Linearization<float>;
