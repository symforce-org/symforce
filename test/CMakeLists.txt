# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests)

# ------------------------------------------------------------------------------
# symforce_test library

file(GLOB_RECURSE SYMFORCE_TEST_SOURCES CONFIGURE_DEPENDS symforce_function_codegen_test_data/**/*.cc)
file(GLOB_RECURSE SYMFORCE_TEST_HEADERS CONFIGURE_DEPENDS symforce_function_codegen_test_data/**/*.hpp)
file(GLOB_RECURSE SYMFORCE_TEST_HEADERS CONFIGURE_DEPENDS include/*.h)
add_library(
  symforce_test
  ${SYMFORCE_LIBRARY_TYPE}
  ${SYMFORCE_TEST_SOURCES}
  ${SYMFORCE_TEST_HEADERS}
)
target_compile_options(symforce_test PRIVATE ${SYMFORCE_COMPILE_OPTIONS})
target_link_libraries(symforce_test Catch2::Catch2WithMain symforce_gen symforce_opt)

# TODO(aaron): Marking these all as system, so we don't warn inside them.  Ideally we'd maybe warn
# in generated code, just not LCM bindings?
target_include_directories(symforce_test
  SYSTEM
  INTERFACE symforce_function_codegen_test_data/symengine/cam_function_codegen_test_data/cpp
  INTERFACE symforce_function_codegen_test_data/symengine/codegen_multi_function_test_data/cpp
  INTERFACE symforce_function_codegen_test_data/symengine/codegen_nan_test_data/cpp
  INTERFACE symforce_function_codegen_test_data/symengine/codegen_sparse_matrix_test_data/cpp
  INTERFACE symforce_function_codegen_test_data/symengine/codegen_cpp_test_data/cpp
  INTERFACE symforce_function_codegen_test_data/sympy/databuffer_codegen_test_data/cpp
  INTERFACE include
  PUBLIC symforce_function_codegen_test_data/symengine/codegen_explicit_template_instantiation_test_data/cpp
)
target_compile_definitions(
  symforce_test
  INTERFACE SYMFORCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/..
)

# ------------------------------------------------------------------------------
# Tests

file(GLOB SYMFORCE_TEST_CC_SOURCES CONFIGURE_DEPENDS *_test.cc)
set(SYMFORCE_CC_TEST_TARGETS "")
foreach(SOURCE_FILE ${SYMFORCE_TEST_CC_SOURCES})
  get_filename_component(TEST_NAME ${SOURCE_FILE} NAME_WE)
  set(BINARY_NAME tests/${TEST_NAME})
  set(TARGET_NAME bin.tests.${TEST_NAME})

  add_executable(${TARGET_NAME} ${SOURCE_FILE})
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${BINARY_NAME})
  set_target_properties(${TARGET_NAME} PROPERTIES TEST_NAME ${TEST_NAME})
  set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  target_link_libraries(
    ${TARGET_NAME}
    symforce_gen
    symforce_opt
    symforce_examples
    symforce_test
    symforce_slam
  )

  list(APPEND SYMFORCE_CC_TEST_TARGETS ${TARGET_NAME})

  add_test(${TEST_NAME} ${CMAKE_BINARY_DIR}/bin/${BINARY_NAME})
endforeach()

if(SYMFORCE_ADD_PYTHON_TESTS)
  file(GLOB SYMFORCE_TEST_PY_SOURCES CONFIGURE_DEPENDS *_test.py)
  foreach(SOURCE_FILE ${SYMFORCE_TEST_PY_SOURCES})
    get_filename_component(TEST_NAME ${SOURCE_FILE} NAME_WE)

    add_test(NAME symengine_${TEST_NAME}
      COMMAND ${CMAKE_COMMAND} -E env SYMFORCE_LOGLEVEL=WARNING SYMFORCE_SYMBOLIC_API=symengine ${SYMFORCE_PYTHON} ${SOURCE_FILE}
    )

    add_test(NAME sympy_${TEST_NAME}
      COMMAND ${CMAKE_COMMAND} -E env SYMFORCE_LOGLEVEL=WARNING SYMFORCE_SYMBOLIC_API=sympy ${SYMFORCE_PYTHON} ${SOURCE_FILE}
    )
  endforeach()
endif()

get_directory_property(have_parent_scope PARENT_DIRECTORY)
if(have_parent_scope)
  set(SYMFORCE_CC_TEST_TARGETS "${SYMFORCE_CC_TEST_TARGETS}" PARENT_SCOPE)
endif()
