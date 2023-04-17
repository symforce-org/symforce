/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./gnc_optimizer.h"

#include "./optimizer.h"

// Explicit instantiation
template class sym::GncOptimizer<sym::Optimizer<double>>;
template class sym::GncOptimizer<sym::Optimizer<float>>;
