/* ----------------------------------------------------------------------------
 * SymForce - Copyright 2022, Skydio, Inc.
 * This source code is under the Apache 2.0 license found in the LICENSE file.
 * ---------------------------------------------------------------------------- */

#include "./assert.h"

#include <sstream>

namespace sym {

std::string FormatFailure(const char* error, const char* func, const char* file, int line) {
  std::stringstream ss;
  ss << "SYM_ASSERT: " << error << std::endl;
  ss << "    --> " << func << std::endl;
  ss << "    --> " << file << ":" << line << std::endl;
  return ss.str();
}

}  // namespace sym
