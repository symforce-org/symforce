set(SRC
    symengine_wrapper.cpp
    pywrapper.cpp
    )

configure_file(config.pxi.in config.pxi)

include_directories(BEFORE ${python_wrapper_BINARY_DIR}/symengine/lib)

cython_add_module_pyx(symengine_wrapper symengine.pxd)
add_python_library(symengine_wrapper ${SRC})
target_link_libraries(symengine_wrapper ${SYMENGINE_LIBRARIES})
if (CMAKE_CXX_COMPILER_ID MATCHES GNU|Clang)
  # Must suppress strict aliasing for this file
  set_source_files_properties(
    symengine_wrapper.cpp
    PROPERTIES COMPILE_FLAGS "-fno-strict-aliasing -Wno-unused-function"
    )
endif()

add_custom_target(cython
    COMMAND cython symengine_wrapper.pyx
    )

set(PY_PATH ${PYTHON_INSTALL_PATH}/symengine/lib)
install(TARGETS symengine_wrapper
            RUNTIME DESTINATION ${PY_PATH}
            ARCHIVE DESTINATION ${PY_PATH}
            LIBRARY DESTINATION ${PY_PATH}
        )
install(FILES __init__.py
    ${CMAKE_CURRENT_BINARY_DIR}/config.pxi
    symengine.pxd
    symengine_wrapper.pxd
    pywrapper.h
    DESTINATION ${PY_PATH}
    )

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
