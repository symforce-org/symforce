project(test_matrix)

add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix symengine catch)
add_test(test_matrix ${PROJECT_BINARY_DIR}/test_matrix)

add_executable(test_matrixexpr test_matrixexpr.cpp)
target_link_libraries(test_matrixexpr symengine catch)
add_test(test_matrixexpr ${PROJECT_BINARY_DIR}/test_matrixexpr)
