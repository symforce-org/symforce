project(test_ntheory)

add_executable(test_ntheory test_ntheory.cpp)
target_link_libraries(test_ntheory symengine catch)
add_test(test_ntheory ${PROJECT_BINARY_DIR}/test_ntheory)

add_executable(test_diophantine test_diophantine.cpp)
target_link_libraries(test_diophantine symengine catch)
add_test(test_diophantine ${PROJECT_BINARY_DIR}/test_diophantine)
