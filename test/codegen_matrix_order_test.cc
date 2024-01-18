/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

#include "symforce_function_codegen_test_data/symengine/codegen_matrix_order_data/matrix_order.h"

TEST_CASE("Codegened matrix order is correct", "[codegen_matrix_order]") {
  const auto expected = (Eigen::Matrix<double, 2, 3>() << 1, 2, 3, 4, 5, 6).finished();

  CHECK(codegen_matrix_order::MatrixOrder<double>() == expected);
}
