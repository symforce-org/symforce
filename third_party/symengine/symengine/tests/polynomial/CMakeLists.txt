project(test_polynomial)

add_executable(test_uintpoly test_uintpoly.cpp)
target_link_libraries(test_uintpoly symengine catch)
add_test(test_uintpoly ${PROJECT_BINARY_DIR}/test_uintpoly)

add_executable(test_uratpoly test_uratpoly.cpp)
target_link_libraries(test_uratpoly symengine catch)
add_test(test_uratpoly ${PROJECT_BINARY_DIR}/test_uratpoly)

add_executable(test_mintpoly test_mintpoly.cpp)
target_link_libraries(test_mintpoly symengine catch)
add_test(test_mintpoly ${PROJECT_BINARY_DIR}/test_mintpoly)

add_executable(test_uexprpoly test_uexprpoly.cpp)
target_link_libraries(test_uexprpoly symengine catch)
add_test(test_uexprpoly ${PROJECT_BINARY_DIR}/test_uexprpoly)

add_executable(test_mexprpoly test_mexprpoly.cpp)
target_link_libraries(test_mexprpoly symengine catch)
add_test(test_mexprpoly ${PROJECT_BINARY_DIR}/test_mexprpoly)

add_executable(test_basic_conversions test_basic_conversions.cpp)
target_link_libraries(test_basic_conversions symengine catch)
add_test(test_basic_conversions ${PROJECT_BINARY_DIR}/test_basic_conversions)

if (WITH_FLINT)
    add_executable(test_uintpoly_flint test_uintpoly_flint.cpp)
    target_link_libraries(test_uintpoly_flint symengine catch)
    add_test(test_uintpoly_flint ${PROJECT_BINARY_DIR}/test_uintpoly_flint)

    add_executable(test_uratpoly_flint test_uratpoly_flint.cpp)
    target_link_libraries(test_uratpoly_flint symengine catch)
    add_test(test_uratpoly_flint ${PROJECT_BINARY_DIR}/test_uratpoly_flint)

    add_executable(test_cancel test_cancel.cpp)
    target_link_libraries(test_cancel symengine catch)
    add_test(test_cancel ${PROJECT_BINARY_DIR}/test_cancel)
endif()

if (WITH_PIRANHA)
    add_executable(test_uintpoly_piranha test_uintpoly_piranha.cpp)
    target_link_libraries(test_uintpoly_piranha symengine catch)
    add_test(test_uintpoly_piranha ${PROJECT_BINARY_DIR}/test_uintpoly_piranha)

    add_executable(test_uratpoly_piranha test_uratpoly_piranha.cpp)
    target_link_libraries(test_uratpoly_piranha symengine catch)
    add_test(test_uratpoly_piranha ${PROJECT_BINARY_DIR}/test_uratpoly_piranha)
endif()
