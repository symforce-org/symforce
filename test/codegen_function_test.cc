#include "../gen/cpp/geo/rot3.h"
#include "./symforce_function_codegen_test_data/codegen_test_cpp_data/codegen_test_cpp/constants_t.hpp"
#include "./symforce_function_codegen_test_data/codegen_test_cpp_data/codegen_test_cpp/states_t.hpp"
#include "./symforce_function_codegen_test_data/codegen_test_cpp_data/codegen_test_cpp.h"

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

int main(int argc, char** argv) {
    double x = 2.0;
    double y = -5.0;
    geo::Rot3<double> rot;
    codegen_test_cpp::states_t states;
    states.p[0] = 1.0;
    states.p[1] = 2.0;
    codegen_test_cpp::constants_t constants;
    constants.epsilon = 1e-8;

    double foo;
    double bar;
    codegen_test_cpp::CodegenTestCpp<double>(x, y, rot, constants, states, &foo, &bar);
    assertTrue(std::abs(foo - (std::pow(x, 2) + rot.Data()[3])) < 1e-8);
    assertTrue(std::abs(bar - (constants.epsilon + std::sin(y) + std::pow(x, 2))) < 1e-8);
}
