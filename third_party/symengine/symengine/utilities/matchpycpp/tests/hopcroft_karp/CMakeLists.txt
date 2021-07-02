project(test_hopcroft_karp)

add_executable(${PROJECT_NAME} test_hopcroft_karp.cpp)
target_link_libraries(${PROJECT_NAME} symengine catch)
add_test(${PROJECT_NAME} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
