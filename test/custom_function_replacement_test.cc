/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "symforce_function_codegen_test_data/sympy/custom_function_replacement.h"

#include <math.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Generated function with function replacement is correct", "[custom_func_replacement]") {
  CHECK(cpp_code_printer_test::TestExpression<double>(0) == 2.0);
  CHECK(cpp_code_printer_test::TestExpression<double>(M_PI) == 0.0);
}
