include(LibFindMacros)

libfind_library(gmp gmp)
set(GMP_LIBRARIES ${GMP_LIBRARY})
set(GMP_TARGETS gmp)

if (WITH_GMPXX)
    libfind_include(gmpxx.h gmp)
    libfind_library(gmpxx gmp)
    set(GMP_LIBRARIES ${GMPXX_LIBRARY} ${GMP_LIBRARIES})
    set(GMP_TARGETS ${GMP_TARGETS} gmpxx)
else()
    libfind_include(gmp.h gmp)
endif()

set(GMP_INCLUDE_DIRS ${GMP_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP DEFAULT_MSG GMP_LIBRARIES
    GMP_INCLUDE_DIRS)

mark_as_advanced(GMP_INCLUDE_DIR GMPXX_LIBRARY GMP_LIBRARY)
