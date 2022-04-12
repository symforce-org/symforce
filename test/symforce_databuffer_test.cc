/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include <cmath>

#include <symforce/buffer_test/buffer_func.h>

#include "catch.hpp"

static constexpr int N_BUFFER = 4;

TEMPLATE_TEST_CASE("Test databuffer func", "[with sympy]", double, float) {
  using Scalar = TestType;

  std::array<Scalar, N_BUFFER> buffer{};

  for (int i = 0; i < N_BUFFER; i++) {
    buffer[i] = i;
  }

  const Scalar a = 1;
  const Scalar b = 2;
  Scalar result = buffer_test::BufferFunc<Scalar>(a, b, buffer.data());
  // expression is buffer[(a + b) * (b - a)] + buffer[b * b - a * a] + (a + b)
  // 2 * buffer[b^2 - a^2] + (a+b)
  // 2 * buffer[3] + 3
  const Scalar expected = 9;

  CHECK(result == Catch::Approx(expected).epsilon(1e-20));
}
