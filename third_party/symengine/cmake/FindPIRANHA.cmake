include(LibFindMacros)

libfind_include(piranha/piranha.hpp piranha)

set(PIRANHA_INCLUDE_DIRS ${PIRANHA_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PIRANHA DEFAULT_MSG
    PIRANHA_INCLUDE_DIRS)

mark_as_advanced(PIRANHA_INCLUDE_DIR)
