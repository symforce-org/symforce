#include <sym/rot3.h>
#include <symforce/codegen_multi_function_test/codegen_multi_function_test1.h>
#include <symforce/codegen_multi_function_test/codegen_multi_function_test2.h>

#include <lcmtypes/codegen_multi_function_test/inputs_constants_t.hpp>
#include <lcmtypes/codegen_multi_function_test/inputs_states_t.hpp>
#include <lcmtypes/codegen_multi_function_test/inputs_t.hpp>
#include <lcmtypes/codegen_multi_function_test/outputs_1_t.hpp>
#include <lcmtypes/codegen_multi_function_test/outputs_2_t.hpp>

#include "catch.hpp"

TEST_CASE("Multi-function codegen compiles", "[codegen_multi_function]") {
  codegen_multi_function_test::inputs_t inputs;
  inputs.x = 2.0;
  inputs.y = -5.0;
  sym::Rot3<double> rot;
  std::copy_n(rot.Data().data(), 4, &inputs.rot[0]);
  inputs.states.p[0] = 1.0;
  inputs.states.p[1] = 2.0;
  inputs.constants.epsilon = 1e-8;

  codegen_multi_function_test::outputs_1_t outputs_1;
  codegen_multi_function_test::CodegenMultiFunctionTest1<double>(inputs, &outputs_1);
  CHECK(outputs_1.foo == Catch::Approx(std::pow(inputs.x, 2) + inputs.rot[3]).epsilon(1e-8));
  CHECK(outputs_1.bar ==
        Catch::Approx(inputs.constants.epsilon + std::sin(inputs.y) + std::pow(inputs.x, 2))
            .epsilon(1e-8));

  codegen_multi_function_test::outputs_2_t outputs_2;
  codegen_multi_function_test::CodegenMultiFunctionTest2<double>(inputs, &outputs_2);
  CHECK(outputs_2.foo == Catch::Approx(std::pow(inputs.y, 3) + inputs.x).epsilon(1e-8));
}
