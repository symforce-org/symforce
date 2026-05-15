if (WITH_SYMENGINE_TEUCHOS)
    add_subdirectory(utilities/teuchos)
    # Include Teuchos headers:
    include_directories(BEFORE ${teuchos_SOURCE_DIR})
    # Include Teuchos config file (generated per build):
    include_directories(BEFORE ${teuchos_BINARY_DIR})
endif()

if (WITH_GENERATE_PARSER)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/parser/parser.tab.cc
                      COMMAND ${BISON_EXECUTABLE} -Wcounterexamples -d parser.yy
                      DEPENDS parser/parser.yy
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/parser
                     )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/parser/tokenizer.cpp
                      COMMAND ${RE2C_EXECUTABLE} -W  --no-generation-date -b tokenizer.re -o tokenizer.cpp
                      DEPENDS parser/tokenizer.re
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/parser
                     )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/parser/sbml/sbml_parser.tab.cc
            COMMAND ${BISON_EXECUTABLE} -d sbml_parser.yy
            DEPENDS parser/sbml/sbml_parser.yy
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/parser/sbml
            )

    add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/parser/sbml/sbml_tokenizer.cpp
            COMMAND ${RE2C_EXECUTABLE} -W  --no-generation-date -b sbml_tokenizer.re -o sbml_tokenizer.cpp
            DEPENDS parser/sbml/sbml_tokenizer.re
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/parser/sbml
            )
endif()

add_subdirectory(utilities/matchpycpp)

set(SRC
    add.cpp
    basic.cpp
    complex.cpp
    complex_double.cpp
    constants.cpp
    cse.cpp
    cwrapper.cpp
    data_buffer_element.cpp
    dense_matrix.cpp
    derivative.cpp
    dict.cpp
    diophantine.cpp
    eval.cpp
    eval_double.cpp
    expand.cpp
    expression.cpp
    fields.cpp
    finitediff.cpp
    functions.cpp
    infinity.cpp
    integer.cpp
    logic.cpp
    matrix.cpp
    monomials.cpp
    mul.cpp
    nan.cpp
    ntheory.cpp
    ntheory_funcs.cpp
    number.cpp
    numer_denom.cpp
    parser/parser_old.cpp
    parser/parser.tab.cc
    parser/parser.cpp
    parser/tokenizer.cpp
    parser/sbml/sbml_parser.tab.cc
    parser/sbml/sbml_parser.cpp
    parser/sbml/sbml_tokenizer.cpp
    polys/basic_conversions.cpp
    polys/msymenginepoly.cpp
    polys/uexprpoly.cpp
    polys/uintpoly.cpp
    polys/uratpoly.cpp
    pow.cpp
    prime_sieve.cpp
    printers/codegen.cpp
    printers/mathml.cpp
    printers/sbml.cpp
    printers/strprinter.cpp
    printers/latex.cpp
    printers/unicode.cpp
    printers/stringbox.cpp
    rational.cpp
    real_double.cpp
    rewrite.cpp
    rings.cpp
    series.cpp
    series_generic.cpp
    sets.cpp
    set_funcs.cpp
    solve.cpp
    sparse_matrix.cpp
    symbol.cpp
    symengine_rcp.cpp
    visitor.cpp
    test_visitors.cpp
    assumptions.cpp
    refine.cpp
    simplify.cpp
    tuple.cpp
    matrices/identity_matrix.cpp
    matrices/matrix_symbol.cpp
    matrices/zero_matrix.cpp
    matrices/diagonal_matrix.cpp
    matrices/immutable_dense_matrix.cpp
    matrices/matrix_add.cpp
    matrices/hadamard_product.cpp
    matrices/matrix_mul.cpp
    matrices/conjugate_matrix.cpp
    matrices/transpose.cpp
    matrices/trace.cpp
    matrices/size.cpp
    matrices/is_zero.cpp
    matrices/is_real.cpp
    matrices/is_symmetric.cpp
    matrices/is_square.cpp
    matrices/is_diagonal.cpp
    matrices/is_lower.cpp
    matrices/is_upper.cpp
    matrices/is_toeplitz.cpp
)

if ("${SYMENGINE_INTEGER_CLASS}" STREQUAL "BOOSTMP")
    set(SRC ${SRC} mp_boost.cpp)
elseif ("${SYMENGINE_INTEGER_CLASS}" STREQUAL "FLINT")
    set(SRC ${SRC} mp_wrapper.cpp)
elseif ("${SYMENGINE_INTEGER_CLASS}" STREQUAL "GMP")
    set(SRC ${SRC} mp_wrapper.cpp)
endif()

if (WITH_MPFR)
    set(SRC ${SRC} eval_mpfr.cpp real_mpfr.cpp)
endif()

if (WITH_MPC)
    set(SRC ${SRC} eval_mpc.cpp complex_mpc.cpp)
endif()

if (WITH_FLINT)
    set(SRC series_flint.cpp ${SRC})
    set(SRC polys/uintpoly_flint.cpp ${SRC})
endif()

if (WITH_ARB)
    set(SRC ${SRC} eval_arb.cpp)
endif()

if (WITH_PIRANHA)
    set(SRC series_piranha.cpp ${SRC})
    set(SRC polys/uintpoly_piranha.cpp ${SRC})
endif()

if (WITH_LLVM)
    set(SRC ${SRC} llvm_double.cpp)
endif()

# Needed for "make install"
set(HEADERS
    add.h
    basic.h
    basic-inl.h
    basic-methods.inc
    complex_double.h
    complex.h
    complex_mpc.h
    constants.h
    cwrapper.h
    data_buffer_element.h
    derivative.h
    dict.h
    diophantine.h
    eval_arb.h
    eval_double.h
    eval.h
    eval_mpc.h
    eval_mpfr.h
    expression.h
    fields.h
    finitediff.h
    flint_wrapper.h
    functions.h
    infinity.h
    integer.h
    lambda_double.h
    llvm_double.h
    logic.h
    matrix.h
    monomials.h
    mp_class.h
    mp_wrapper.h
    mul.h
    nan.h
    ntheory.h
    ntheory_funcs.h
    number.h
    parser.h
    parser/parser.h
    parser/tokenizer.h
    parser/sbml/sbml_parser.h
    parser/sbml/sbml_tokenizer.h
    polys/basic_conversions.h
    polys/cancel.h
    polys/uexprpoly.h
    polys/uintpoly_flint.h
    polys/uintpoly.h
    polys/uintpoly_piranha.h
    polys/upolybase.h
    polys/uratpoly.h
    polys/usymenginepoly.h
    polys/msymenginepoly.h
    pow.h
    prime_sieve.h
    printers/codegen.h
    printers/mathml.h
    printers/sbml.h
    printers/strprinter.h
    printers/latex.h
    printers/unicode.h
    printers/stringbox.h
    printers.h
    rational.h
    real_double.h
    real_mpfr.h
    rings.h
    serialize-cereal.h
    series_flint.h
    series_generic.h
    series.h
    series_piranha.h
    series_visitor.h
    sets.h
    solve.h
    subs.h
    symbol.h
    symengine_assert.h
    symengine_casts.h
    symengine_exception.h
    symengine_rcp.h
    tribool.h
    type_codes.inc
    visitor.h
    test_visitors.h
    assumptions.h
    refine.h
    simplify.h
    utilities/stream_fmt.h
    tuple.h
    matrix_expressions.h
    matrices/matrix_expr.h
    matrices/identity_matrix.h
    matrices/matrix_symbol.h
    matrices/zero_matrix.h
    matrices/diagonal_matrix.h
    matrices/immutable_dense_matrix.h
    matrices/matrix_add.h
    matrices/hadamard_product.h
    matrices/matrix_mul.h
    matrices/conjugate_matrix.h
    matrices/size.h
    matrices/transpose.h
    matrices/trace.h
)

if (NOT WITH_SYSTEM_CEREAL)
  file(GLOB_RECURSE CEREAL_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "utilities/cereal/include/*")
  set(HEADERS ${HEADERS} ${CEREAL_HEADERS})
endif()

include(GNUInstallDirs)

# Configure SymEngine using our CMake options:
set(SYMENGINE_CLING_LIBRARY_DIR "\"${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\"")

configure_file(symengine_config.h.in symengine_config.h)
configure_file(symengine_config_cling.h.in symengine_config_cling.h)

# Include the config file:
include_directories(BEFORE ${symengine_BINARY_DIR})
# Include the source directory
include_directories(BEFORE ${symengine_SOURCE_DIR})

add_library(symengine ${SRC})

include(GenerateExportHeader)
generate_export_header(symengine)
set_target_properties(symengine PROPERTIES COMPILE_DEFINITIONS "symengine_EXPORTS")
if (BUILD_SHARED_LIBS)
    set_target_properties(
        symengine
        PROPERTIES
        VERSION ${SYMENGINE_VERSION}
        SOVERSION ${SYMENGINE_MAJOR_VERSION}.${SYMENGINE_MINOR_VERSION}
    )
    if (WIN32)
        set_target_properties(
            symengine
            PROPERTIES
            RUNTIME_OUTPUT_NAME symengine-${SYMENGINE_MAJOR_VERSION}.${SYMENGINE_MINOR_VERSION}
        )
    endif()
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_target_properties(symengine PROPERTIES COMPILE_FLAGS "-Wconversion -Wno-sign-conversion")
    set_source_files_properties(parser/parser.tab.cc PROPERTIES COMPILE_FLAGS "-Wno-conversion")
endif()

if (WITH_LLVM)
    if (NOT ${LLVM_VERSION} VERSION_LESS "16.0.0")
        set_target_properties(symengine PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF CXX_STANDARD_REQUIRED yes)
    elseif (NOT ${LLVM_VERSION} VERSION_LESS "10.0.0")
        set_target_properties(symengine PROPERTIES CXX_STANDARD 14 CXX_EXTENSIONS OFF CXX_STANDARD_REQUIRED yes)
    endif()
endif()

if (WITH_SYMENGINE_TEUCHOS)
    target_link_libraries(symengine teuchos)
endif()
target_link_libraries(symengine ${LIBS})

if (WITH_LLVM AND NOT SYMENGINE_LLVM_LINK_DOWNSTREAM AND NOT WITH_SYMENGINE_TEUCHOS)
    if (${EXCLUDE_LIBS})
        set_property(TARGET symengine APPEND_STRING PROPERTY LINK_FLAGS "-Wl,--exclude-libs,ALL")
    elseif (${UNEXPORTED_SYMBOL})
        set(LLVM_SYMBOLS_AVOID "${LLVM_SYMBOLS_AVOID} -Wl,-unexported_symbol,__ZTVN4llvm* -Wl,-unexported_symbol,__ZNK4llvm*")
        set(LLVM_SYMBOLS_AVOID "${LLVM_SYMBOLS_AVOID} -Wl,-unexported_symbol,__ZN4llvm* -Wl,-unexported_symbol,__ZTIN4llvm*")
        set(LLVM_SYMBOLS_AVOID "${LLVM_SYMBOLS_AVOID} -Wl,-unexported_symbol,_LLVM* -Wl,-unexported_symbol,__ZTSN4llvm*")
        set_property(TARGET symengine APPEND_STRING PROPERTY LINK_FLAGS "${LLVM_SYMBOLS_AVOID}")
    endif()
endif()

install(TARGETS symengine
            EXPORT SymEngineTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
install(FILES
    "${symengine_BINARY_DIR}/symengine/symengine_config.h"
    "${symengine_BINARY_DIR}/symengine/symengine_config_cling.h"
    "${symengine_BINARY_DIR}/symengine/symengine_export.h"
        DESTINATION
    include/symengine)

# make install
macro(INSTALL_HEADERS_WITH_DIRECTORY HEADER_LIST)
    FOREACH(HEADER ${${HEADER_LIST}})
        STRING(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
        install(FILES ${HEADER} DESTINATION include/symengine/${DIR})
    ENDFOREACH(HEADER HEADER_LIST)
endmacro(INSTALL_HEADERS_WITH_DIRECTORY)

INSTALL_HEADERS_WITH_DIRECTORY(HEADERS)

if (BUILD_TESTS)
    add_subdirectory(utilities/catch)
    add_subdirectory(tests)
    add_subdirectory(utilities/matchpycpp/tests)
    add_subdirectory(utilities/matchpycpp/autogen_tests)
endif()

if (WITH_COTIRE)
    # throws if CMAKE_VERSION < 2.8.12
    include(cotire)
    cotire(symengine)
endif()
