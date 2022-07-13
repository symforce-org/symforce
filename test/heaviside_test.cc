/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "symforce_function_codegen_test_data/sympy/heaviside.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Generated heaviside function is correct", "[heaviside]") {
  CHECK(cpp_code_printer_test::Heaviside<double>(-1) == 0);
  CHECK(cpp_code_printer_test::Heaviside<double>(1) == 1);
}
