project(test_finitediff)

add_executable(${PROJECT_NAME} test_finitediff.cpp)
target_link_libraries(${PROJECT_NAME} symengine catch)
add_test(${PROJECT_NAME} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
