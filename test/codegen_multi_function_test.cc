/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <lcmtypes/codegen_multi_function_test/inputs_constants_t.hpp>
#include <lcmtypes/codegen_multi_function_test/inputs_states_t.hpp>
#include <lcmtypes/codegen_multi_function_test/inputs_t.hpp>
#include <lcmtypes/codegen_multi_function_test/outputs_1_t.hpp>
#include <lcmtypes/codegen_multi_function_test/outputs_2_t.hpp>

#include <sym/rot3.h>
#include <symforce/codegen_multi_function_test/codegen_multi_function_test1.h>
#include <symforce/codegen_multi_function_test/codegen_multi_function_test2.h>

template <typename T>
void FillChunkOfValues(T& values) {
  sym::Rot3<double> rot;

  values.x = 2.0;
  values.y = -5.0;

  const auto fill_rot = [&rot](eigen_lcm::Vector4d& rot_to_fill) {
    std::copy_n(rot.Data().data(), 4, rot_to_fill.data());
  };

  fill_rot(values.rot);

  for (int i = 0; i < 3; i++) {
    fill_rot(values.rot_vec[i]);
  }

  for (int i = 0; i < 3; i++) {
    values.scalar_vec[i] = i;
  }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        values.list_of_lists[i][j][k] = i + j + k;
      }
    }
  }
}

TEST_CASE("Multi-function codegen compiles", "[codegen_multi_function]") {
  codegen_multi_function_test::inputs_t inputs;

  FillChunkOfValues(inputs);

  for (int i = 0; i < 3; i++) {
    FillChunkOfValues(inputs.values_vec[i]);
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 1; j++) {
      FillChunkOfValues(inputs.values_vec_2D[i][j]);
    }
  }

  inputs.constants.epsilon = 1e-8;

  inputs.states.p[0] = 1.0;
  inputs.states.p[1] = 2.0;

  inputs.big_matrix = Eigen::MatrixXd::Constant(5, 5, 55.5);
  inputs.small_matrix = Eigen::Matrix4d::Constant(44.4);

  codegen_multi_function_test::outputs_1_t outputs_1;
  outputs_1.big_matrix_from_small_matrix = Eigen::MatrixXd::Zero(5, 5);
  codegen_multi_function_test::CodegenMultiFunctionTest1<double>(inputs, &outputs_1);
  CHECK(outputs_1.foo == Catch::Approx(std::pow(inputs.x, 2) + inputs.rot[3]).epsilon(1e-8));
  CHECK(outputs_1.bar ==
        Catch::Approx(inputs.constants.epsilon + std::sin(inputs.y) + std::pow(inputs.x, 2))
            .epsilon(1e-8));
  Eigen::MatrixXd expected_big_matrix_from_small_matrix = Eigen::MatrixXd::Zero(5, 5);
  expected_big_matrix_from_small_matrix.block<4, 4>(0, 0) = inputs.small_matrix;
  CHECK(
      outputs_1.big_matrix_from_small_matrix.isApprox(expected_big_matrix_from_small_matrix, 1e-8));
  CHECK(outputs_1.small_matrix_from_big_matrix.isApprox(inputs.big_matrix.block<4, 4>(0, 0), 1e-8));

  codegen_multi_function_test::outputs_2_t outputs_2;
  codegen_multi_function_test::CodegenMultiFunctionTest2<double>(inputs, &outputs_2);
  CHECK(outputs_2.foo == Catch::Approx(std::pow(inputs.y, 3) + inputs.x).epsilon(1e-8));
}
