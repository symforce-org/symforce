/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <catch2/catch_test_macros.hpp>

#include "symforce_function_codegen_test_data/sympy/custom_function_replacement.h"

TEST_CASE("Generated function with function replacement is correct", "[custom_func_replacement]") {
  const double x = 0.1;  // Test value
  CHECK(cpp_code_printer_test::TestExpression<double>(x) == fast_math::sin<double>(x));
}
