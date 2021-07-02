include(LibFindMacros)

libfind_include(execinfo.h execinfo)
set(EXECINFO_INCLUDE_DIRS ${EXECINFO_INCLUDE_DIR})
include(FindPackageHandleStandardArgs)

if (CMAKE_SYSTEM_NAME MATCHES "BSD")
    libfind_library(execinfo execinfo)
    set(EXECINFO_LIBRARIES ${EXECINFO_LIBRARY})
    set(EXECINFO_TARGETS execinfo)
    find_package_handle_standard_args(EXECINFO DEFAULT_MSG
        EXECINFO_LIBRARIES EXECINFO_INCLUDE_DIRS)
else ()
    find_package_handle_standard_args(EXECINFO DEFAULT_MSG
        EXECINFO_INCLUDE_DIRS)
endif ()

mark_as_advanced(EXECINFO_INCLUDE_DIR EXECINFO_LIBRARY)
