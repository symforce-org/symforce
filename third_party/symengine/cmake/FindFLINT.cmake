include(LibFindMacros)

libfind_include(flint/flint.h flint)
libfind_library(flint flint)

set(FLINT_LIBRARIES ${FLINT_LIBRARY})
set(FLINT_INCLUDE_DIRS ${FLINT_INCLUDE_DIR})
set(FLINT_TARGETS flint)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FLINT DEFAULT_MSG FLINT_LIBRARIES
    FLINT_INCLUDE_DIRS)

mark_as_advanced(FLINT_INCLUDE_DIR FLINT_LIBRARY)
