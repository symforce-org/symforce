/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <lcmtypes/codegen_cpp_test/constants_t.hpp>
#include <lcmtypes/codegen_cpp_test/states_t.hpp>
#include <lcmtypes/codegen_cpp_test/values_vec_t.hpp>

#include <sym/rot3.h>
#include <symforce/codegen_cpp_test/codegen_cpp_test.h>

TEST_CASE("Generated C++ compiles", "[codegen_cpp_test]") {
  double x = 2.0;
  double y = -5.0;
  sym::Rot3<double> rot{};
  std::array<sym::Rot3<double>, 3> rot_vec{};
  std::array<double, 3> scalar_vec{};
  std::array<std::array<sym::Rot3<double>, 3>, 3> list_of_lists{};
  std::array<codegen_cpp_test::values_vec_t, 3> values_vec{};
  std::array<std::array<codegen_cpp_test::values_vec_t, 1>, 2> values_vec_2D{};
  const Eigen::Matrix<double, 5, 5> big_matrix = Eigen::Matrix<double, 5, 5>::Zero();
  codegen_cpp_test::states_t states{};
  states.p[0] = 1.0;
  states.p[1] = 2.0;
  codegen_cpp_test::constants_t constants{};
  constants.epsilon = 1e-8;

  double foo;
  double bar;
  std::array<double, 3> scalar_vec_out;
  std::array<codegen_cpp_test::values_vec_t, 3> values_vec_out;
  std::array<std::array<codegen_cpp_test::values_vec_t, 1>, 2> values_vec_2D_out;

  codegen_cpp_test::CodegenCppTest<double>(
      x, y, rot, rot_vec, scalar_vec, list_of_lists, values_vec, values_vec_2D, constants,
      big_matrix, states, &foo, &bar, &scalar_vec_out, &values_vec_out, &values_vec_2D_out);
  CHECK(foo == Catch::Approx(std::pow(x, 2) + rot.Data()[3]).epsilon(1e-8));
  CHECK(bar == Catch::Approx(constants.epsilon + std::sin(y) + std::pow(x, 2)).epsilon(1e-8));
  // TODO(nathan): Check other outputs (just checking that things compile for now)
}
