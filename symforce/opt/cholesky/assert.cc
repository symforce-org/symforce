/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the LGPL license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./assert.h"

#include <sstream>

namespace sym {
namespace internal {

std::string FormatFailure(const char* error, const char* func, const char* file, int line) {
  std::stringstream ss;
  ss << "SPARSE_MATH_ASSERT: " << error << std::endl;
  ss << "    --> " << func << std::endl;
  ss << "    --> " << file << ":" << line << std::endl;
  return ss.str();
}

}  // namespace internal
}  // namespace sym
