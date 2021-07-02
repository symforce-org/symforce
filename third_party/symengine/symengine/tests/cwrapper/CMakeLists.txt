project(test_cwrapper)

add_executable(${PROJECT_NAME} test_cwrapper.c)
target_link_libraries(${PROJECT_NAME} symengine)
add_test(${PROJECT_NAME} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
