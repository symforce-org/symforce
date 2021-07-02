include(LibFindMacros)

libfind_include(primesieve.hpp primesieve)
libfind_library(primesieve primesieve)

set(PRIMESIEVE_LIBRARIES ${PRIMESIEVE_LIBRARY})
set(PRIMESIEVE_INCLUDE_DIRS ${PRIMESIEVE_INCLUDE_DIR})
set(PRIMESIEVE_TARGETS primesieve)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PRIMESIEVE DEFAULT_MSG
    PRIMESIEVE_LIBRARIES PRIMESIEVE_INCLUDE_DIRS)

mark_as_advanced(PRIMESIEVE_INCLUDE_DIR PRIMESIEVE_LIBRARY)
