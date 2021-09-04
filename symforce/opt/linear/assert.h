#pragma once

#include <string>

namespace math {

/**
 * Format an assertion failure.
 */
std::string FormatFailure(const char* error, const char* func, const char* file, int line);

}  // namespace math

/**
 * Assert a runtime condition with a #define disable mechanism.
 *
 * Cloned from symforce/opt/assert.h, original inspiration taken from:
 *     http://cnicholson.net/2009/02/stupid-c-tricks-adventures-in-assert/
 *
 * TODO(hayk): Improve with custom string, _EQ variant, etc.
 */
#ifndef SPARSE_MATH_DISABLE_ASSERT
#define SPARSE_MATH_ASSERT(expr)                                                  \
  do {                                                                            \
    if (!(expr)) {                                                                \
      throw std::runtime_error(                                                   \
          math::FormatFailure((#expr), __PRETTY_FUNCTION__, __FILE__, __LINE__)); \
    }                                                                             \
  } while (0)
#else
#define SPARSE_MATH_ASSERT(expr) \
  do {                           \
    (void)sizeof(expr);          \
  } while (0)
#endif
