# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

function(add_matrix_multiplication_benchmark matrix_name)
    add_executable(
        matrix_multiplication_benchmark_${matrix_name}
        matrix_multiplication/gen/matrix_multiplication_benchmark_${matrix_name}.cc
    )

    target_link_libraries(
        matrix_multiplication_benchmark_${matrix_name}
        symforce_catch2
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

# TODO(aaron): These take forever to compile, fix that and re-enable
# add_matrix_multiplication_benchmark(rotor1)
# add_matrix_multiplication_benchmark(ww_36_pmec_36)
# add_matrix_multiplication_benchmark(mcca)

# -----------------------------------------------------------------------------

add_executable(
    inverse_compose_jacobian_benchmark
    inverse_compose_jacobian/inverse_compose_jacobian_benchmark.cc
)

find_package(Sophus REQUIRED)
target_link_libraries(
    inverse_compose_jacobian_benchmark
    gtsam
    symforce_catch2
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
    symforce_catch2
    symforce_gen
    symforce_opt
    symforce_examples
    Ceres::ceres
)

set_target_properties(robot_3d_localization_benchmark
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)
