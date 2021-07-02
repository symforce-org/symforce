include(LibFindMacros)

libfind_include(mpfr.h mpfr)
libfind_library(mpfr mpfr)

set(MPFR_LIBRARIES ${MPFR_LIBRARY})
set(MPFR_INCLUDE_DIRS ${MPFR_INCLUDE_DIR})
set(MPFR_TARGETS mpfr)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFR DEFAULT_MSG MPFR_LIBRARIES
    MPFR_INCLUDE_DIRS)

mark_as_advanced(MPFR_INCLUDE_DIR MPFR_LIBRARY)
