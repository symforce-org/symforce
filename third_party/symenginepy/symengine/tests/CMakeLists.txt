set(PY_PATH ${PYTHON_INSTALL_PATH}/symengine/tests)
install(FILES __init__.py
    test_arit.py
    test_dict_basic.py
    test_eval.py
    test_expr.py
    test_functions.py
    test_number.py
    test_matrices.py
    test_ntheory.py
    test_printing.py
    test_sage.py
    test_series_expansion.py
    test_sets.py
    test_solve.py
    test_subs.py
    test_symbol.py
    test_sympify.py
    test_sympy_conv.py
    test_var.py
    test_lambdify.py
    test_sympy_compat.py
    test_logic.py
    DESTINATION ${PY_PATH}
    )
