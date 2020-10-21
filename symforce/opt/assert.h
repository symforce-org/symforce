#pragma once

#include <string>

namespace sym {

/**
 * Format an assertion failure.
 */
std::string FormatFailure(const char* error, const char* func, const char* file, int line);

}  // namespace sym

/**
 * Assert a runtime condition with a #define disable mechanism.
 *
 * Inspiration taken from:
 *     http://cnicholson.net/2009/02/stupid-c-tricks-adventures-in-assert/
 *
 * TODO(hayk): Improve with custom string, _EQ variant, etc.
 */
#ifndef SYMFORCE_DISABLE_ASSERT
#define SYM_ASSERT(expr)                                                         \
  do {                                                                           \
    if (!(expr)) {                                                               \
      throw std::runtime_error(                                                  \
          sym::FormatFailure((#expr), __PRETTY_FUNCTION__, __FILE__, __LINE__)); \
    }                                                                            \
  } while (0)
#else
#define SYM_ASSERT(expr) \
  do {                   \
    (void)sizeof(expr);  \
  } while (0)
#endif
