project(test_bipartite)

add_executable(${PROJECT_NAME} test_bipartite.cpp)
target_link_libraries(${PROJECT_NAME} symengine catch)
add_test(${PROJECT_NAME} ${PROJECT_NAME})
