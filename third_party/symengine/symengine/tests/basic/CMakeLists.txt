project(test_basic)

add_executable(${PROJECT_NAME} test_basic.cpp)
target_link_libraries(${PROJECT_NAME} symengine catch)
add_test(${PROJECT_NAME} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})

add_executable(test_arit test_arit.cpp)
target_link_libraries(test_arit symengine catch)
add_test(test_arit ${PROJECT_BINARY_DIR}/test_arit)

add_executable(test_poly test_poly.cpp)
target_link_libraries(test_poly symengine catch)
add_test(test_poly ${PROJECT_BINARY_DIR}/test_poly)

add_executable(test_series test_series.cpp)
target_link_libraries(test_series symengine catch)
add_test(test_series ${PROJECT_BINARY_DIR}/test_series)

add_executable(test_series_generic test_series_generic.cpp)
target_link_libraries(test_series_generic symengine catch)
add_test(test_series_generic ${PROJECT_BINARY_DIR}/test_series_generic)

if (WITH_PIRANHA)
    add_executable(test_series_expansion_UP test_series_expansion_UP.cpp)
    target_link_libraries(test_series_expansion_UP symengine catch)
    add_test(test_series_expansion_UP ${PROJECT_BINARY_DIR}/test_series_expansion_UP)

    add_executable(test_series_expansion_URatP test_series_expansion_URatP.cpp)
    target_link_libraries(test_series_expansion_URatP symengine catch)
    add_test(test_series_expansion_URatP ${PROJECT_BINARY_DIR}/test_series_expansion_URatP)
endif()

if (WITH_FLINT)
    add_executable(test_series_expansion_URatF test_series_expansion_URatF.cpp)
    target_link_libraries(test_series_expansion_URatF symengine catch)
    add_test(test_series_expansion_URatF ${PROJECT_BINARY_DIR}/test_series_expansion_URatF)
endif()

add_executable(test_functions test_functions.cpp)
target_link_libraries(test_functions symengine catch)
add_test(test_functions ${PROJECT_BINARY_DIR}/test_functions)

add_executable(test_subs test_subs.cpp)
target_link_libraries(test_subs symengine catch)
add_test(test_subs ${PROJECT_BINARY_DIR}/test_subs)

add_executable(test_integer test_integer.cpp)
target_link_libraries(test_integer symengine catch)
add_test(test_integer ${PROJECT_BINARY_DIR}/test_integer)

add_executable(test_rational test_rational.cpp)
target_link_libraries(test_rational symengine catch)
add_test(test_rational ${PROJECT_BINARY_DIR}/test_rational)

add_executable(test_relationals test_relationals.cpp)
target_link_libraries(test_relationals symengine catch)
add_test(test_relationals ${PROJECT_BINARY_DIR}/test_relationals)

add_executable(test_number test_number.cpp)
target_link_libraries(test_number symengine catch)
add_test(test_number ${PROJECT_BINARY_DIR}/test_number)

add_executable(test_as_numer_denom test_as_numer_denom.cpp)
target_link_libraries(test_as_numer_denom symengine catch)
add_test(test_as_numer_denom ${PROJECT_BINARY_DIR}/test_as_numer_denom)

add_executable(test_parser test_parser.cpp)
target_link_libraries(test_parser symengine catch)
add_test(test_parser ${PROJECT_BINARY_DIR}/test_parser)

add_executable(test_serialize-cereal test_serialize-cereal.cpp)
target_link_libraries(test_serialize-cereal symengine catch)
add_test(test_serialize-cereal ${PROJECT_BINARY_DIR}/test_serialize-cereal)

add_executable(test_sbml_parser test_sbml_parser.cpp)
target_link_libraries(test_sbml_parser symengine catch)
add_test(test_sbml_parser ${PROJECT_BINARY_DIR}/test_sbml_parser)

add_executable(test_sets test_sets.cpp)
target_link_libraries(test_sets symengine catch)
add_test(test_sets ${PROJECT_BINARY_DIR}/test_sets)

add_executable(test_fields test_fields.cpp)
target_link_libraries(test_fields symengine catch)
add_test(test_fields ${PROJECT_BINARY_DIR}/test_fields)

add_executable(test_infinity test_infinity.cpp)
target_link_libraries(test_infinity symengine catch)
add_test(test_infinity ${PROJECT_BINARY_DIR}/test_infinity)

add_executable(test_nan test_nan.cpp)
target_link_libraries(test_nan symengine catch)
add_test(test_nan ${PROJECT_BINARY_DIR}/test_nan)

add_executable(test_solve test_solve.cpp)
target_link_libraries(test_solve symengine catch)
add_test(test_solve ${PROJECT_BINARY_DIR}/test_solve)

add_executable(test_as_real_imag test_as_real_imag.cpp)
target_link_libraries(test_as_real_imag symengine catch)
add_test(test_as_real_imag ${PROJECT_BINARY_DIR}/test_as_real_imag)

if ("${INTEGER_CLASS}" MATCHES "boostmp")
	add_executable(test_integer_class test_integer_class.cpp)
	target_link_libraries(test_integer_class symengine catch)
	add_test(test_integer_class ${PROJECT_BINARY_DIR}/test_integer_class)
endif()

add_executable(test_cse test_cse.cpp)
target_link_libraries(test_cse symengine catch)
add_test(test_cse ${PROJECT_BINARY_DIR}/test_cse)


add_executable(test_count_ops test_count_ops.cpp)
target_link_libraries(test_count_ops symengine catch)
add_test(test_count_ops ${PROJECT_BINARY_DIR}/test_count_ops)

add_executable(test_test_visitors test_test_visitors.cpp)
target_link_libraries(test_test_visitors symengine catch)
add_test(test_test_visitors ${PROJECT_BINARY_DIR}/test_test_visitors)

add_executable(test_assumptions test_assumptions.cpp)
target_link_libraries(test_assumptions symengine catch)
add_test(test_assumptions ${PROJECT_BINARY_DIR}/test_assumptions)

add_executable(test_refine test_refine.cpp)
target_link_libraries(test_refine symengine catch)
add_test(test_refine ${PROJECT_BINARY_DIR}/test_refine)

add_executable(test_simplify test_simplify.cpp)
target_link_libraries(test_simplify symengine catch)
add_test(test_simplify ${PROJECT_BINARY_DIR}/test_simplify)

add_executable(test_tuple test_tuple.cpp)
target_link_libraries(test_tuple symengine catch)
add_test(test_tuple ${PROJECT_BINARY_DIR}/test_tuple)

add_executable(test_tribool test_tribool.cpp)
target_link_libraries(test_tribool symengine catch)
add_test(test_tribool ${PROJECT_BINARY_DIR}/test_tribool)
