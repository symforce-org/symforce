#include <iostream>
#include <math.h>

#include <sym/rot3.h>
#include <symforce/codegen_nan_test/identity_dist_jacobian.h>

#include "catch.hpp"

TEST_CASE("Codegen function does not generate NaN", "[codegen_nan_test]") {
  std::cout << "*** Testing codegen function for NaNs ***" << std::endl;

  sym::Rot3<double> rot = sym::Rot3<double>::Identity();
  double epsilon = 1e-6;
  double res = codegen_nan_test::IdentityDistJacobian<double>(rot, epsilon);
  CHECK_FALSE(std::isnan(res));
}
