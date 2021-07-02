project(test_logic)

add_executable(test_logic test_logic.cpp)
target_link_libraries(test_logic symengine catch)
add_test(test_logic ${PROJECT_BINARY_DIR}/test_logic)
