#include "./assert.h"

#include <sstream>

namespace math {

std::string FormatFailure(const char* error, const char* func, const char* file, int line) {
  std::stringstream ss;
  ss << "SPARSE_MATH_ASSERT: " << error << std::endl;
  ss << "    --> " << func << std::endl;
  ss << "    --> " << file << ":" << line << std::endl;
  return ss.str();
}

}  // namespace math
