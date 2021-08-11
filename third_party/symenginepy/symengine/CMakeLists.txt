add_subdirectory(lib)
add_subdirectory(tests)

set(PY_PATH ${PYTHON_INSTALL_PATH}/symengine)
install(FILES __init__.py utilities.py sympy_compat.py functions.py printing.py
    DESTINATION ${PY_PATH}
    )
