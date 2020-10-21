#include <geo/rot3.h>
#include <symforce/codegen_multi_function/codegen_multi_function_test1.h>
#include <symforce/codegen_multi_function/codegen_multi_function_test2.h>

#include <lcmtypes/codegen_multi_function/inputs_constants_t.hpp>
#include <lcmtypes/codegen_multi_function/inputs_states_t.hpp>
#include <lcmtypes/codegen_multi_function/inputs_t.hpp>
#include <lcmtypes/codegen_multi_function/outputs_1_t.hpp>
#include <lcmtypes/codegen_multi_function/outputs_2_t.hpp>

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

int main(int argc, char** argv) {
  codegen_multi_function::inputs_t inputs;
  inputs.x = 2.0;
  inputs.y = -5.0;
  geo::Rot3<double> rot;
  std::copy_n(rot.Data().data(), 4, &inputs.rot[0]);
  inputs.states.p[0] = 1.0;
  inputs.states.p[1] = 2.0;
  inputs.constants.epsilon = 1e-8;

  codegen_multi_function::outputs_1_t outputs_1;
  codegen_multi_function::CodegenMultiFunctionTest1<double>(inputs, &outputs_1);
  assertTrue(std::abs(outputs_1.foo - (std::pow(inputs.x, 2) + inputs.rot[3])) < 1e-8);
  assertTrue(std::abs(outputs_1.bar - (inputs.constants.epsilon + std::sin(inputs.y) +
                                       std::pow(inputs.x, 2))) < 1e-8);

  codegen_multi_function::outputs_2_t outputs_2;
  codegen_multi_function::CodegenMultiFunctionTest2<double>(inputs, &outputs_2);
  assertTrue(std::abs(outputs_2.foo - (std::pow(inputs.y, 3) + inputs.x)) < 1e-8);
}
