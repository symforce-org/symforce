#include "../gen/cpp/geo/rot3.h"
#include "./symforce_function_codegen_test_data/codegen_multi_function_ns/inputs_t.h"
#include "./symforce_function_codegen_test_data/codegen_multi_function_ns/outputs_1_t.h"
#include "./symforce_function_codegen_test_data/codegen_multi_function_ns/outputs_2_t.h"
#include "./symforce_function_codegen_test_data/codegen_multi_function_ns/inputs_constants_t.h"
#include "./symforce_function_codegen_test_data/codegen_multi_function_ns/inputs_states_t.h"
#include "./symforce_function_codegen_test_data/codegen_multi_function_test1.h"
#include "./symforce_function_codegen_test_data/codegen_multi_function_test2.h"

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

int main(int argc, char** argv) {
    codegen_multi_function_ns::inputs_t inputs;
    inputs.x = 2.0;
    inputs.y = -5.0;
    geo::Rot3<double> rot;
    std::copy_n(rot.Data().data(), inputs.rot.size(), inputs.rot.begin());
    inputs.states.p = {1.0, 2.0};
    inputs.constants.epsilon = 1e-8;

    codegen_multi_function_ns::outputs_1_t outputs_1;
    codegen_multi_function_ns::CodegenMultiFunctionTest1<double>(inputs, &outputs_1);
    assertTrue(std::abs(outputs_1.foo - (std::pow(inputs.x, 2) + inputs.rot[3])) < 1e-8);
    assertTrue(std::abs(outputs_1.bar - (inputs.constants.epsilon + std::sin(inputs.y) + std::pow(inputs.x, 2))) < 1e-8);

    codegen_multi_function_ns::outputs_2_t outputs_2;
    codegen_multi_function_ns::CodegenMultiFunctionTest2<double>(inputs, &outputs_2);
    assertTrue(std::abs(outputs_2.foo - (std::pow(inputs.y, 3) + inputs.x)) < 1e-8);
}
