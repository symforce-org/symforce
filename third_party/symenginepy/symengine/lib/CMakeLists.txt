set(SRC
    ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.cpp
    pywrapper.cpp
    )

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pxd
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/symengine_wrapper.in.pxd
       ${PROJECT_SOURCE_DIR}/cmake/preprocess.py
    COMMAND ${PYTHON_BIN} ${PROJECT_SOURCE_DIR}/cmake/preprocess.py
       ${CMAKE_CURRENT_SOURCE_DIR}/symengine_wrapper.in.pxd
       ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pxd
       HAVE_SYMENGINE_MPFR=${HAVE_SYMENGINE_MPFR}
       HAVE_SYMENGINE_MPC=${HAVE_SYMENGINE_MPC}
       HAVE_SYMENGINE_PIRANHA=${HAVE_SYMENGINE_PIRANHA}
       HAVE_SYMENGINE_FLINT=${HAVE_SYMENGINE_FLINT}
       HAVE_SYMENGINE_LLVM=${HAVE_SYMENGINE_LLVM}
       HAVE_SYMENGINE_LLVM_LONG_DOUBLE=${HAVE_SYMENGINE_LLVM_LONG_DOUBLE}
    COMMENT "Preprocessing symengine_wrapper.in.pxd"
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pyx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/symengine_wrapper.in.pyx
       ${CMAKE_CURRENT_SOURCE_DIR}/symengine_wrapper.in.pxd
       ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pxd
       ${PROJECT_SOURCE_DIR}/cmake/preprocess.py
    COMMAND ${PYTHON_BIN} ${PROJECT_SOURCE_DIR}/cmake/preprocess.py
       ${CMAKE_CURRENT_SOURCE_DIR}/symengine_wrapper.in.pyx
       ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pyx
       HAVE_SYMENGINE_MPFR=${HAVE_SYMENGINE_MPFR}
       HAVE_SYMENGINE_MPC=${HAVE_SYMENGINE_MPC}
       HAVE_SYMENGINE_PIRANHA=${HAVE_SYMENGINE_PIRANHA}
       HAVE_SYMENGINE_FLINT=${HAVE_SYMENGINE_FLINT}
       HAVE_SYMENGINE_LLVM=${HAVE_SYMENGINE_LLVM}
       HAVE_SYMENGINE_LLVM_LONG_DOUBLE=${HAVE_SYMENGINE_LLVM_LONG_DOUBLE}
    COMMENT "Preprocessing symengine_wrapper.in.pyx"
)

cython_add_module_pyx(symengine_wrapper.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pyx
    ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pxd
    ${CMAKE_CURRENT_SOURCE_DIR}/symengine.pxd)
add_python_library(symengine_wrapper ${SRC})
target_link_libraries(symengine_wrapper ${SYMENGINE_LIBRARIES})
if (CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
  # Must suppress strict aliasing for this file
  set_source_files_properties(
    symengine_wrapper.cpp
    PROPERTIES COMPILE_FLAGS "-fno-strict-aliasing -Wno-unused-function"
    )
endif()

set(PY_PATH ${PYTHON_INSTALL_PATH}/symengine/lib)
install(TARGETS symengine_wrapper
            RUNTIME DESTINATION ${PY_PATH}
            ARCHIVE DESTINATION ${PY_PATH}
            LIBRARY DESTINATION ${PY_PATH}
        )
install(FILES
    symengine.pxd
    ${CMAKE_CURRENT_BINARY_DIR}/symengine_wrapper.pxd
    pywrapper.h
    DESTINATION ${PY_PATH}
    )

if (SYMENGINE_INSTALL_PY_FILES)
  install(FILES __init__.py DESTINATION ${PY_PATH})
endif ()

if (${SYMENGINE_COPY_EXTENSION})
  if ("${PYTHON_EXTENSION_SOABI}" MATCHES "ppc64le")
    string(REPLACE "ppc64le" "powerpc64le" COPY_PYTHON_EXTENSION_SOABI "${PYTHON_EXTENSION_SOABI}")
  endif ()
  if ("${PYTHON_EXTENSION_SOABI}" MATCHES "powerpc64le")
    string(REPLACE "powerpc64le" "ppc64le" COPY_PYTHON_EXTENSION_SOABI "${PYTHON_EXTENSION_SOABI}")
  endif ()
  message("${PYTHON_EXTENSION_SOABI} ${COPY_PYTHON_EXTENSION_SOABI}")
  set(SOURCE_NAME "${PY_PATH}/symengine_wrapper${PYTHON_EXTENSION_SOABI}.so")
  set(DEST_NAME "${PY_PATH}/symengine_wrapper${COPY_PYTHON_EXTENSION_SOABI}.so")
  install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_NAME} ${DEST_NAME})")
endif ()
