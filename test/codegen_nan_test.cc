/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <math.h>

#include <catch2/catch_test_macros.hpp>

#include <sym/rot3.h>
#include <symforce/codegen_nan_test/identity_dist_jacobian.h>

TEST_CASE("Codegen function does not generate NaN", "[codegen_nan_test]") {
  sym::Rot3<double> rot = sym::Rot3<double>::Identity();
  double epsilon = 1e-6;
  double res = codegen_nan_test::IdentityDistJacobian<double>(rot, epsilon);
  CAPTURE(res);
  CHECK_FALSE(std::isnan(res));
}
