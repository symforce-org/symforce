include(LibFindMacros)

libfind_include(pthread.h pthread)
libfind_library(pthread pthread)

set(PTHREAD_LIBRARIES ${PTHREAD_LIBRARY})
set(PTHREAD_INCLUDE_DIRS ${PTHREAD_INCLUDE_DIR})
set(PTHREAD_TARGETS pthread)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PTHREAD DEFAULT_MSG PTHREAD_LIBRARIES
    PTHREAD_INCLUDE_DIRS)

mark_as_advanced(PTHREAD_INCLUDE_DIR PTHEARD_LIBRARY)
