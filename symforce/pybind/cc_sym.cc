/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <pybind11/pybind11.h>

#include "./cc_factor.h"
#include "./cc_key.h"
#include "./cc_linearization.h"
#include "./cc_logger.h"
#include "./cc_optimization_stats.h"
#include "./cc_optimizer.h"
#include "./cc_values.h"

PYBIND11_MODULE(cc_sym, generated_module) {
  generated_module.doc() = "This module wraps many of the C++ optimization classes.";

  // NOTE(aaron): The ordering matters here, because pybind11-stubgen needs types used in signatures
  // to be declared before the function is defined
  sym::AddKeyWrapper(generated_module);
  sym::AddValuesWrapper(generated_module);
  sym::AddFactorWrapper(generated_module);
  sym::AddLinearizationWrapper(generated_module);
  sym::AddOptimizationStatsWrapper(generated_module);
  sym::AddOptimizerWrapper(generated_module);
  sym::AddLoggerWrapper(generated_module);
}
