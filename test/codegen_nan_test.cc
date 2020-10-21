#include <iostream>
#include <math.h>

#include <geo/rot3.h>
#include <symforce/codegen_nan_test/identity_dist_jacobian.h>

// TODO(hayk): Use the catch unit testing framework (single header).
#define assertTrue(a)                                      \
  if (!(a)) {                                              \
    std::ostringstream o;                                  \
    o << __FILE__ << ":" << __LINE__ << ": Test failure."; \
    throw std::runtime_error(o.str());                     \
  }

int main(int argc, char** argv) {
  std::cout << "*** Testing codegen function for NaNs ***" << std::endl;

  geo::Rot3<double> rot = geo::Rot3<double>::Identity();
  double epsilon = 1e-6;
  double res = codegen_nan_test::IdentityDistJacobian<double>(rot, epsilon);
  assertTrue(!std::isnan(res));
}
