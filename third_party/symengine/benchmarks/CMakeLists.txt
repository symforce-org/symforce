project(benchmarks)

include_directories(BEFORE ${symengine_SOURCE_DIR})
include_directories(BEFORE ${symengine_BINARY_DIR})
include_directories(BEFORE ${teuchos_SOURCE_DIR})
include_directories(BEFORE ${teuchos_BINARY_DIR})

add_executable(expand1 expand1.cpp)
target_link_libraries(expand1 symengine)

add_executable(eval_double1 eval_double1.cpp)
target_link_libraries(eval_double1 symengine)

add_executable(expand2 expand2.cpp)
target_link_libraries(expand2 symengine)

add_executable(expand2b expand2b.cpp)
target_link_libraries(expand2b symengine)

add_executable(expand3 expand3.cpp)
target_link_libraries(expand3 symengine)

add_executable(add1 add1.cpp)
target_link_libraries(add1 symengine)

add_executable(matrix_add1 matrix_add1.cpp)
target_link_libraries(matrix_add1 symengine)

add_executable(matrix_add2 matrix_add2.cpp)
target_link_libraries(matrix_add2 symengine)

add_executable(matrix_mul1 matrix_mul1.cpp)
target_link_libraries(matrix_mul1 symengine)

add_executable(matrix_mul2 matrix_mul2.cpp)
target_link_libraries(matrix_mul2 symengine)

add_executable(symbench symbench.cpp)
target_link_libraries(symbench symengine)

add_executable(lwbench lwbench.cpp)
target_link_libraries(lwbench symengine)

add_executable(expand6 expand6.cpp)
target_link_libraries(expand6 symengine)

add_executable(expand6b expand6b.cpp)
target_link_libraries(expand6b symengine)

add_executable(expand7 expand7.cpp)
target_link_libraries(expand7 symengine)

add_executable(series series.cpp)
target_link_libraries(series symengine)

add_executable(symengine_bench symengine_bench.cpp)
target_link_libraries(symengine_bench symengine)

add_executable(ntheorybench ntheorybench.cpp)
target_link_libraries(ntheorybench symengine)

if (WITH_FLINT)
    add_executable(series_expansion_sincos_flint series_expansion_sincos_flint.cpp)
    target_link_libraries(series_expansion_sincos_flint symengine)
endif()

if (WITH_PIRANHA)
    add_executable(series_expansion_sincos_piranha series_expansion_sincos_piranha.cpp)
    target_link_libraries(series_expansion_sincos_piranha symengine)

    add_executable(series_expansion_sinp series_expansion_sinp.cpp)
    target_link_libraries(series_expansion_sinp symengine)
endif()

if (BUILD_BENCHMARKS_GOOGLE)
    find_package(benchmark REQUIRED)

    add_executable(bench bench.cpp)
    target_link_libraries(bench symengine benchmark::benchmark)

    add_executable(bench_eval_double bench_eval_double.cpp)
    target_link_libraries(bench_eval_double symengine benchmark::benchmark)

    add_executable(visitor_call visitor_call.cpp)
    target_link_libraries(visitor_call symengine benchmark::benchmark)

    add_executable(visitor_init visitor_init.cpp)
    target_link_libraries(visitor_init symengine benchmark::benchmark)

    add_executable(parsing_google parsing_google.cpp)
    target_link_libraries(parsing_google symengine benchmark::benchmark)
endif()

add_executable(parsing parsing.cpp)
target_link_libraries(parsing symengine)

add_executable(diff_cache diff_cache.cpp)
target_link_libraries(diff_cache symengine)
