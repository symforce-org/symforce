/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <Eigen/Dense>
#include <catch2/catch_test_macros.hpp>

#include <lcmtypes/codegen_explicit_template_instantiation_test/constants_t.hpp>
#include <lcmtypes/codegen_explicit_template_instantiation_test/states_t.hpp>
#include <lcmtypes/codegen_explicit_template_instantiation_test/values_vec_t.hpp>

#include <symforce/codegen_explicit_template_instantiation_test/codegen_explicit_template_instantiation_test.h>

template <typename Scalar>
void ExplicitTemplateInstantiationTestHelper() {
  // Helper function for calling "ExplicitTemplateInstantiationTest" with different scalar types

  // Fill inputs
  Scalar x = 1;
  Scalar y = 2;
  sym::Rot3<Scalar> rot{};
  std::array<sym::Rot3<Scalar>, 3> rot_vec{};
  std::array<Scalar, 3> scalar_vec{};
  std::array<std::array<sym::Rot3<Scalar>, 3>, 3> list_of_lists{};
  std::array<codegen_explicit_template_instantiation_test::values_vec_t, 3> values_vec{};
  std::array<std::array<codegen_explicit_template_instantiation_test::values_vec_t, 1>, 2>
      values_vec_2D{};
  codegen_explicit_template_instantiation_test::constants_t constants{};
  const Eigen::Matrix<Scalar, 5, 5> big_matrix = Eigen::Matrix<Scalar, 5, 5>::Zero();
  codegen_explicit_template_instantiation_test::states_t states{};

  // Initialize outputs
  Scalar foo;
  Scalar bar;
  std::array<Scalar, 3> scalar_vec_out;
  std::array<codegen_explicit_template_instantiation_test::values_vec_t, 3> values_vec_out;
  std::array<std::array<codegen_explicit_template_instantiation_test::values_vec_t, 1>, 2>
      values_vec_2D_out;

  CodegenExplicitTemplateInstantiationTest(
      x, y, rot, rot_vec, scalar_vec, list_of_lists, values_vec, values_vec_2D, constants,
      big_matrix, states, &foo, &bar, &scalar_vec_out, &values_vec_out, &values_vec_2D_out);
}

TEST_CASE("Test explicit template instantiation", "[codegen_test]") {
  // This test is used to check that generated function which have explicit template instantiation
  // can be compiled an called without errors.

  CHECK_NOTHROW(ExplicitTemplateInstantiationTestHelper<double>());
  CHECK_NOTHROW(ExplicitTemplateInstantiationTestHelper<float>());
}
