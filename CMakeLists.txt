project(symforce)
# NOTE(aaron): This is the minimum version for FetchContent, not sure if we need newer; certainly no
# newer than 3.15 required
cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# User-Configurable Options
# ==============================================================================

# ------------------------------------------------------------------------------
# Enable/Disable Components

option(SYMFORCE_BUILD_GEN_ONLY "Build only the symforce_gen target" OFF)
option(SYMFORCE_BUILD_EXAMPLES "Build the examples" ON)
option(SYMFORCE_BUILD_TESTS "Build the tests" ON)
option(SYMFORCE_BUILD_SYMENGINE "Build symengine[py]" ON)

# ------------------------------------------------------------------------------
# TicTocs

option(SYMFORCE_CUSTOM_TIC_TOCS
  "Use the user-provided TicTocs instead of the builtin implementation"
  OFF
)
set(SYMFORCE_TIC_TOC_TARGET "" CACHE STRING
  "If SYMFORCE_CUSTOM_TIC_TOCS is on, this should be a target to depend on that provides TicTocs"
)
set(SYMFORCE_TIC_TOC_HEADER "" CACHE STRING
  "If SYMFORCE_CUSTOM_TIC_TOCS is on, this should be the name of a header to include that provides the TicTocs implementation"
)
set(SYMFORCE_TIC_TOC_MACRO "" CACHE STRING
  "If SYMFORCE_CUSTOM_TIC_TOCS is on, this should be the implementation of the SYM_TIME_SCOPE macro (typically another macro, or a function)"
)

# ------------------------------------------------------------------------------
# Misc

option(SYMFORCE_BUILD_STATIC_LIBRARIES
  "Build libraries as static (symforce_opt_linear is always built as shared)"
  OFF
)

set(SYMFORCE_PYTHON "python3" CACHE STRING "Python executable to use")

set(SYMFORCE_SYMENGINE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/symengine_install CACHE STRING
  "Directory to install symengine"
)

# ==============================================================================
# Setup
# ==============================================================================

if(SYMFORCE_BUILD_STATIC_LIBRARIES)
  set(SYMFORCE_LIBRARY_TYPE STATIC)
else()
  set(SYMFORCE_LIBRARY_TYPE SHARED)
endif()

# ==============================================================================
# Third Party Dependencies (needed for build)
# ==============================================================================

include(FetchContent)

# ------------------------------------------------------------------------------
# eigen3

find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found, adding with FetchContent")
  FetchContent_Declare(
    eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen
    GIT_TAG 3147391d946bb4b6c68edd901f2add6ac1f31f8c  # release 3.4.0
  )
  FetchContent_MakeAvailable(eigen3)
endif()

# These dependencies are not needed for symforce_gen
if(NOT SYMFORCE_BUILD_GEN_ONLY)
  # ------------------------------------------------------------------------------
  # fmtlib

  find_package(fmt QUIET)
  if (NOT fmt_FOUND)
    message(STATUS "fmt not found, adding with FetchContent")
    FetchContent_Declare(
      fmtlib
      GIT_REPOSITORY https://github.com/fmtlib/fmt
      GIT_TAG d141cdbeb0fb422a3fb7173b285fd38e0d1772dc # release 8.0.1
    )
    FetchContent_MakeAvailable(fmtlib)
  else()
    message(STATUS "fmt found")
  endif()

  # ------------------------------------------------------------------------------
  # spdlog

  find_package(spdlog QUIET)
  if (NOT spdlog_FOUND)
    message(STATUS "spdlog not found, adding with FetchContent")
    set(SPDLOG_FMT_EXTERNAL ON CACHE INTERNAL "spdlog shouldn't use its bundled fmtlib")
    FetchContent_Declare(
      spdlog
      GIT_REPOSITORY https://github.com/gabime/spdlog
      GIT_TAG eb3220622e73a4889eee355ffa37972b3cac3df5 # release 1.9.2
    )
    FetchContent_MakeAvailable(spdlog)
  else()
    message(STATUS "spdlog found")
  endif()

  # ------------------------------------------------------------------------------
  # METIS

  # Libs for metis - our version of metis does not bundle this
  FetchContent_Declare(
    GKlib
    GIT_REPOSITORY https://github.com/KarypisLab/GKlib
    # GKlib does not have releases, this is picked to be around our METIS version
    GIT_TAG 3c4d4853957fd23d93e608c826acfbb3d6b8941e
    # GKlib adds -march=native on gcc, which is incorrect if cross-compiling.  This cmake file is
    # also included directly in METIS, so we can't just overwrite GKlib_COPTIONS, we have to patch
    PATCH_COMMAND sed -i "${FETCHCONTENT_BASE_DIR}/gklib-src/GKlibSystem.cmake" -e "s/-march=native//"
  )
  FetchContent_MakeAvailable(GKlib)

  function(add_metis)
    FetchContent_Declare(
      metis
      GIT_REPOSITORY https://github.com/KarypisLab/METIS
      # METIS on GitHub doesn't build with straight CMake after 81d70ba, and between c903288 and
      # 81d70ba it tries and fails to include GKlib as a submodule.  So we use c903288, and add GKlib
      # ourselves (above).
      GIT_TAG c9032886f6073478d1013b4b2d75d59071530394
    )

    # Tell metis where to find GKlib
    FetchContent_GetProperties(GKlib SOURCE_DIR GKlib_SOURCE_DIR)
    set(GKLIB_PATH ${GKlib_SOURCE_DIR} CACHE PATH "path to GKlib" FORCE)

    FetchContent_MakeAvailable(metis)

    # libmetis has warnings on gcc 4.6+, but they add -Werror, so we need to cancel that out
    target_compile_options(metis PRIVATE "-Wno-error")

    # Metis doesn't put its main header (metis.h) on the metis target
    FetchContent_GetProperties(metis SOURCE_DIR metis_SOURCE_DIR)
    target_include_directories(metis INTERFACE ${metis_SOURCE_DIR}/include)
  endfunction()

  add_metis()
endif(NOT SYMFORCE_BUILD_GEN_ONLY)

# ==============================================================================
# SymForce Targets
# ==============================================================================

# ------------------------------------------------------------------------------
# symforce_gen

file(GLOB_RECURSE SYMFORCE_GEN_SOURCES CONFIGURE_DEPENDS gen/cpp/**/*.cc)
file(GLOB_RECURSE SYMFORCE_GEN_HEADERS CONFIGURE_DEPENDS gen/cpp/**/*.tcc gen/cpp/**/*.h)
add_library(
  symforce_gen
  ${SYMFORCE_LIBRARY_TYPE}
  ${SYMFORCE_GEN_SOURCES}
  ${SYMFORCE_GEN_HEADERS}
)
target_link_libraries(symforce_gen Eigen3::Eigen)
target_include_directories(
  symforce_gen
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/gen/cpp
)

if(NOT SYMFORCE_BUILD_GEN_ONLY)
  # ------------------------------------------------------------------------------
  # symforce_opt_linear

  # NOTE(hayk): For licensing reasons keep this as a shared library and
  # do not link statically. Treat this as code for which we can provide source.
  file(GLOB SYMFORCE_OPT_LINEAR_SOURCES CONFIGURE_DEPENDS symforce/opt/linear/*.cc)
  file(GLOB SYMFORCE_OPT_LINEAR_HEADERS CONFIGURE_DEPENDS symforce/opt/linear/*.h symforce/opt/linear/*.tcc)
  add_library(
    symforce_opt_linear
    SHARED
    ${SYMFORCE_OPT_LINEAR_SOURCES}
    ${SYMFORCE_OPT_LINEAR_HEADERS}
  )
  target_link_libraries(symforce_opt_linear metis Eigen3::Eigen)
  target_include_directories(symforce_opt_linear PUBLIC .)

  # ------------------------------------------------------------------------------
  # symforce_opt

  file(GLOB_RECURSE SYMFORCE_OPT_SOURCES CONFIGURE_DEPENDS symforce/opt/*.cc symforce/opt/**/*.cc)
  file(GLOB_RECURSE SYMFORCE_OPT_HEADERS CONFIGURE_DEPENDS symforce/opt/*.h symforce/opt/**/*.h symforce/opt/*.tcc symforce/opt/**/*.tcc)
  add_library(
    symforce_opt
    ${SYMFORCE_LIBRARY_TYPE}
    ${SYMFORCE_OPT_SOURCES}
    ${SYMFORCE_OPT_HEADERS}
  )
  target_link_libraries(symforce_opt
    symforce_gen
    symforce_opt_linear
    fmt::fmt
    spdlog::spdlog
    Eigen3::Eigen
  )

  if(SYMFORCE_CUSTOM_TIC_TOCS)
    target_link_libraries(symforce_opt ${SYMFORCE_TIC_TOC_TARGET})
    target_compile_definitions(symforce_opt
      PUBLIC SYMFORCE_TIC_TOC_HEADER=${SYMFORCE_TIC_TOC_HEADER}
      PUBLIC SYM_TIME_SCOPE=${SYMFORCE_TIC_TOC_MACRO}
    )
  endif(SYMFORCE_CUSTOM_TIC_TOCS)
endif(NOT SYMFORCE_BUILD_GEN_ONLY)

# ==============================================================================
# Examples and Tests
# ==============================================================================

if(SYMFORCE_BUILD_EXAMPLES)
  # ------------------------------------------------------------------------------
  # symforce_examples

  file(GLOB_RECURSE SYMFORCE_EXAMPLES_SOURCES CONFIGURE_DEPENDS symforce/examples/**/*.cc)
  file(GLOB_RECURSE SYMFORCE_EXAMPLES_HEADERS CONFIGURE_DEPENDS symforce/examples/**/*.h symforce/examples/**/*.tcc)
  add_library(
    symforce_examples
    ${SYMFORCE_LIBRARY_TYPE}
    ${SYMFORCE_EXAMPLES_SOURCES}
    ${SYMFORCE_EXAMPLES_HEADERS}
  )
  target_link_libraries(
    symforce_examples
    symforce_gen
    symforce_opt
    Eigen3::Eigen
  )
  target_include_directories(
    symforce_examples
    PUBLIC symforce/examples/bundle_adjustment_fixed_size/gen/cpp
  )
endif()

if(SYMFORCE_BUILD_TESTS)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/tests)

  # ------------------------------------------------------------------------------
  # symforce_test library

  # TODO(aaron): This goes away when we upgrade to Catch2 v3
  add_library(
    symforce_catch2
    ${SYMFORCE_LIBRARY_TYPE}
    third_party/catch2/src/catch_amalgamated.cpp test/catch_main.cc
  )
  target_include_directories(
    symforce_catch2
    PUBLIC third_party/catch2/include
  )

  add_library(symforce_test INTERFACE)
  target_link_libraries(symforce_test INTERFACE symforce_catch2)
  target_include_directories(symforce_test
    INTERFACE test/symforce_function_codegen_test_data/symengine/cam_function_codegen_test_data/cpp
    INTERFACE test/symforce_function_codegen_test_data/symengine/codegen_multi_function_test_data/cpp
    INTERFACE test/symforce_function_codegen_test_data/symengine/codegen_nan_test_data/cpp
    INTERFACE test/symforce_function_codegen_test_data/symengine/codegen_sparse_matrix_test_data/cpp
    INTERFACE test/symforce_function_codegen_test_data/symengine/codegen_cpp_test_data/cpp
  )
  target_compile_definitions(
    symforce_test
    INTERFACE SYMFORCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
  )

  # ------------------------------------------------------------------------------
  # C++ tests

  enable_testing()
  add_library(symforce_tests_metadata INTERFACE)
  file(GLOB SYMFORCE_TEST_CC_SOURCES CONFIGURE_DEPENDS test/*_test.cc)
  set(SYMFORCE_CC_TEST_TARGETS "")
  foreach(SOURCE_FILE ${SYMFORCE_TEST_CC_SOURCES})
    get_filename_component(TEST_NAME ${SOURCE_FILE} NAME_WE)
    set(BINARY_NAME tests/${TEST_NAME})
    set(TARGET_NAME bin.tests.${TEST_NAME})

    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${BINARY_NAME})
    set_target_properties(${TARGET_NAME} PROPERTIES TEST_NAME ${TEST_NAME})
    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    target_link_libraries(
      ${TARGET_NAME}
      symforce_gen
      symforce_opt
      symforce_examples
      symforce_test
    )

    list(APPEND SYMFORCE_CC_TEST_TARGETS ${TARGET_NAME})

    add_test(${TEST_NAME} ${CMAKE_BINARY_DIR}/bin/${BINARY_NAME})
  endforeach()

  get_directory_property(have_parent_scope PARENT_DIRECTORY)
  if(have_parent_scope)
    set(SYMFORCE_CC_TEST_TARGETS "${SYMFORCE_CC_TEST_TARGETS}" PARENT_SCOPE)
  endif()

endif(SYMFORCE_BUILD_TESTS)

# ==============================================================================
# Third Party Dependencies (not needed for build)
# ==============================================================================

if(SYMFORCE_BUILD_SYMENGINE)

  # ------------------------------------------------------------------------------
  # SymEngine
  include(ExternalProject)
  ExternalProject_Add(symengine
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/symengine
    INSTALL_DIR ${SYMFORCE_SYMENGINE_INSTALL_PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
               -DBUILD_TESTS=OFF
               -DBUILD_BENCHMARKS=OFF
  )

  # ------------------------------------------------------------------------------
  # symenginepy

  ExternalProject_Add(symenginepy
    DEPENDS symengine
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/symenginepy
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${SYMFORCE_PYTHON} ${PROJECT_SOURCE_DIR}/third_party/symenginepy/setup.py
        --verbose
        build_ext
        --build-type=Release
        --symengine-dir ${SYMFORCE_SYMENGINE_INSTALL_PREFIX}/lib/
    INSTALL_COMMAND ${SYMFORCE_PYTHON} ${PROJECT_SOURCE_DIR}/third_party/symenginepy/setup.py
        --verbose
        install
        --prefix ${SYMFORCE_SYMENGINE_INSTALL_PREFIX}
  )

  # Rebuild symengine[py] on changed files
  function(add_sourcechanged_dependency project_name)
    set(cmake_stampdir ${CMAKE_CURRENT_BINARY_DIR}/${project_name}-prefix/src/${project_name}-stamp)
    set(git_stampfile ${CMAKE_CURRENT_BINARY_DIR}/${project_name}.stamp)
    set(git_check_rulename ${project_name}_check_git_clean)
    add_custom_target(${git_check_rulename}
      COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rerun_if_needed.py
                  --path_to_check ${PROJECT_SOURCE_DIR}/third_party/${project_name}
                  --stamp_file ${git_stampfile}
                  --command_to_run "rm -rf ${cmake_stampdir}"
      )

    add_dependencies(${project_name} ${git_check_rulename})
  endfunction()

  add_sourcechanged_dependency(symengine)
  add_sourcechanged_dependency(symenginepy)

endif(SYMFORCE_BUILD_SYMENGINE)