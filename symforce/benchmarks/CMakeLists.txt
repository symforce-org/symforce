# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

# ==============================================================================
# Third Party Dependencies
# ==============================================================================

include(FetchContent)

# ------------------------------------------------------------------------------
# Sophus

find_package(Sophus QUIET)
if (NOT Sophus_FOUND)
  message(STATUS "Sophus not found, adding with FetchContent")
  function(add_sophus)
    set(BUILD_TESTS OFF CACHE INTERNAL "Sophus shouldn't build tests by default")
    set(CMAKE_POSITION_INDEPENDENT_CODE True)
    FetchContent_Declare(
      sophus
      URL https://github.com/strasdat/Sophus/archive/refs/tags/v1.0.0.zip
      URL_HASH SHA256=60c8a18f750fdfd75c828253c8f123fac15493ced0baaee8a19df64a4acbf548
    )
    FetchContent_MakeAvailable(sophus)
    find_package(Sophus)
  endfunction()

  add_sophus()
else()
  message(STATUS "Sophus found")
endif()

# ------------------------------------------------------------------------------
# GTSAM

find_package(gtsam QUIET)
if (NOT gtsam_FOUND)
  # NOTE(aaron): gtsam will not build against a copy of Eigen we've downloaded but not installed,
  # because they include the eigen include directories in their install targets, and CMake
  # complains (correctly) about directories inside the build folder in install targets.  They
  # handle this correctly for their bundled version, but we cannot use that because the version of
  # Eigen that GTSAM is built with must match our version (which is what's used when we call GTSAM
  # code)
  message(STATUS "GTSAM not found, adding with FetchContent")
  function(add_gtsam)
    set(GTSAM_BUILD_WITH_MARCH_NATIVE OFF CACHE INTERNAL "GTSAM shouldn't build native by default")
    set(GTSAM_BUILD_TESTS OFF CACHE INTERNAL "Don't build GTSAM tests")
    set(GTSAM_BUILD_EXAMPLES_ALWAYS OFF CACHE INTERNAL "Don't build GTSAM examples")
    set(GTSAM_WITH_TBB OFF CACHE INTERNAL "Don't build TBB with GTSAM for accurate profiling")
    # NOTE(aaron): This seems to have no effect on the experiments we're doing
    set(GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR ON CACHE INTERNAL "Use the correct BetweenFactor")
    # NOTE(aaron): This has no effect other than making inverse compose chained slower
    set(GTSAM_USE_QUATERNIONS ON CACHE INTERNAL "Use quaternions for Rot3")
    set(CMAKE_POSITION_INDEPENDENT_CODE True)
    FetchContent_Declare(
      gtsam
      URL https://github.com/borglab/gtsam/archive/refs/tags/4.1.1.zip
      URL_HASH SHA256=f03cd72884647aaa0a24df5ee262ad0824b343554c763d81b2957836f4cab99d
    )
    FetchContent_MakeAvailable(gtsam)
  endfunction()

  add_gtsam()
else()
  message(STATUS "GTSAM found")
endif()

# ------------------------------------------------------------------------------
# Ceres

find_package(Ceres QUIET)
if (NOT Ceres_FOUND)
  message(STATUS "Ceres not found, adding with FetchContent")
  function(add_ceres)
    set(BUILD_TESTING OFF CACHE INTERNAL "Don't enable tests")
    set(BUILD_EXAMPLES OFF CACHE INTERNAL "Don't build examples")
    set(BUILD_BENCHMARKS OFF CACHE INTERNAL "Don't build Ceres benchmarking suite")
    set(PROVIDE_UNINSTALL_TARGET
      OFF CACHE INTERNAL
      "Ceres shouldn't add uninstall target, gtsam already adds a target with the same name that collides"
      FORCE
    )
    set(
      CERES_THREADING_MODEL "NO_THREADS" CACHE INTERNAL "Don't use threads for benchmarking" FORCE
    )
    set(CMAKE_POSITION_INDEPENDENT_CODE True)
    FetchContent_Declare(
      ceres
      URL https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.0.0.zip
      URL_HASH SHA256=db12d37b4cebb26353ae5b7746c7985e00877baa8e7b12dc4d3a1512252fff3b
    )
    FetchContent_MakeAvailable(ceres)
  endfunction()

  add_ceres()
else()
  message(STATUS "Ceres found")
endif()

# ==============================================================================
# Benchmark Targets
# ==============================================================================

function(add_matrix_multiplication_benchmark matrix_name)
    add_executable(
        matrix_multiplication_benchmark_${matrix_name}
        matrix_multiplication/gen/matrix_multiplication_benchmark_${matrix_name}.cc
    )

    target_link_libraries(
        matrix_multiplication_benchmark_${matrix_name}
        Catch2::Catch2WithMain
        symforce_gen
        symforce_opt
    )

    set_target_properties(matrix_multiplication_benchmark_${matrix_name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
    )
endfunction()

add_matrix_multiplication_benchmark(b1_ss)
add_matrix_multiplication_benchmark(Tina_DisCog)
add_matrix_multiplication_benchmark(n3c4_b2)
add_matrix_multiplication_benchmark(bibd_9_3)
add_matrix_multiplication_benchmark(lp_sc105)
add_matrix_multiplication_benchmark(rotor1)

# -----------------------------------------------------------------------------

add_executable(
    inverse_compose_jacobian_benchmark
    inverse_compose_jacobian/inverse_compose_jacobian_benchmark.cc
)

find_package(Sophus REQUIRED)
target_link_libraries(
    inverse_compose_jacobian_benchmark
    gtsam
    Catch2::Catch2WithMain
    symforce_gen
    symforce_opt
)
target_include_directories(inverse_compose_jacobian_benchmark PRIVATE ${Sophus_INCLUDE_DIR})

set_target_properties(inverse_compose_jacobian_benchmark
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)

# -----------------------------------------------------------------------------

add_executable(
    robot_3d_localization_benchmark
    robot_3d_localization/robot_3d_localization_benchmark.cc
)

target_link_libraries(
    robot_3d_localization_benchmark
    gtsam
    Catch2::Catch2WithMain
    symforce_gen
    symforce_opt
    symforce_examples
    Ceres::ceres
)

set_target_properties(robot_3d_localization_benchmark
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)
