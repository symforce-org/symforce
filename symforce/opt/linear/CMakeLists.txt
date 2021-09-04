mc_declare_module(DISABLE_CODE_LINTING)

# Include METIS for sparse solver ordering
# TODO(hayk): I guess GTSAM appends the build type to its libraries. Resolve
# this and get rid of this if hell.
set(METIS_INCLUDE_DIR
   ${CMAKE_SOURCE_DIR}/third_party_modules/gtsam_module/gtsam/gtsam/3rdparty/metis/include)
if(BUILD_MC_CC_LIBRARY_AS_STATIC)
    if (${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
        set(METIS_LIB ${CMAKE_BINARY_DIR}/lib/libmetisRelWithDebInfo.a)
    else()
        set(METIS_LIB ${CMAKE_BINARY_DIR}/lib/libmetis.a)
    endif()
else()
    if (${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
        set(METIS_LIB ${CMAKE_BINARY_DIR}/lib/libmetisRelWithDebInfo.so)
    else()
        set(METIS_LIB ${CMAKE_BINARY_DIR}/lib/libmetis.so)
    endif()
endif()

# NOTE(hayk): For licensing reasons keep this as a shared library and
# do not link statically. Treat this as code for which we can provide source.
mc_cc_library(${MODULE_NAME}
        SHARED
        SOURCES *.cc
        HEADERS *.h
        INCLUDE_DIRS PUBLIC ${METIS_INCLUDE_DIR}
        LIB_DEPENDS
            ${METIS_LIB}
        PKG_DEPENDS
            eigen3
        IN_TARGETSETS
            vehicle
            workstation
)

mc_cc_add_test_dir(
        TARGET_DEPENDS ${MODULE_NAME}
            common
            error_reporter
            gtest_util
            math_shared
            tic_toc
        IN_TARGETSETS
            vehicle
            workstation
)
