#include "symforce_function_codegen_test_data/sympy/heaviside.h"

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

int main(int argc, char** argv) {
  assertTrue(cpp_code_printer_test::Heaviside(-1) == 0);
  assertTrue(cpp_code_printer_test::Heaviside(1) == 1);
}
