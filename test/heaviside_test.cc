#include "symforce_function_codegen_test_data/sympy/heaviside.h"

#include "catch.hpp"

TEST_CASE("Generated heaviside function is correct", "[heaviside]") {
  CHECK(cpp_code_printer_test::Heaviside<double>(-1) == 0);
  CHECK(cpp_code_printer_test::Heaviside<double>(1) == 1);
}
