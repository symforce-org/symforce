include(LibFindMacros)

libfind_include(cereal/types/vector.hpp cereal)

set(CEREAL_INCLUDE_DIRS ${CEREAL_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CEREAL DEFAULT_MSG CEREAL_INCLUDE_DIRS)

mark_as_advanced(CEREAL_INCLUDE_DIR)
