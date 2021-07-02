# Macros for finding external libraries
#
# Example usage:
#
#     include(LibFindMacros)
#     libfind_include(gmpxx.h gmp)
#     libfind_library(gmpxx gmp)
#     libfind_library(gmp gmp)
#     set(GMP_LIBRARIES ${GMPXX_LIBRARY} ${GMP_LIBRARY})
#     set(GMP_INCLUDE_DIRS ${GMP_INCLUDE_DIR})
#     include(FindPackageHandleStandardArgs)
#     find_package_handle_standard_args(GMP DEFAULT_MSG GMP_LIBRARIES
#         GMP_INCLUDE_DIRS)
#     mark_as_advanced(GMP_INCLUDE_DIR GMPXX_LIBRARY GMP_LIBRARY)
#
# The result of the Find*.cmake (e.g. FindGMP.cmake) module should be two
# variables GMP_LIBRARIES and GMP_INCLUDE_DIRS, that the user then uses in the
# following way:
#
#     find_package(GMP REQUIRED)
#     include_directories(${GMP_INCLUDE_DIRS})
#     set(LIBS ${LIBS} ${GMP_LIBRARIES})
#     # LIBS is later used in target_link_libraries()

function (libfind_library libname pkg)
    string(TOUPPER ${pkg} PKG)
    string(TOUPPER ${libname} LIBNAME)

    find_library(${LIBNAME}_LIBRARY
        NAMES
            ${libname}
    )

    if (NOT TARGET ${libname})
        add_library(${libname} UNKNOWN IMPORTED)
        set_property(TARGET ${libname} PROPERTY IMPORTED_LOCATION ${${LIBNAME}_LIBRARY})
    endif()
endfunction()

function (libfind_include HEADER pkg)
    string(TOUPPER ${pkg} PKG)

    find_path(${PKG}_INCLUDE_DIR
        NAMES
            ${HEADER}
    )
endfunction()
