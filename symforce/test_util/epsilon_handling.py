# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import symforce
import symforce.symbolic as sf
from symforce import typing as T

try:
    # Attempt, works if we're in ipython
    _default_display_func = display  # type: ignore
except NameError:
    _default_display_func = print


def _limit_and_simplify(
    expr: sf.Expr, x: sf.Scalar, value: sf.Scalar, limit_direction: str
) -> sf.Scalar:
    return sf.simplify(expr.limit(x, value, limit_direction))


def is_value_with_epsilon_correct(
    func: T.Callable[[sf.Scalar, sf.Scalar], sf.Expr],
    singularity: sf.Scalar = 0,
    limit_direction: str = "+",
    display_func: T.Callable[[T.Any], None] = _default_display_func,
    expected_value: sf.Scalar = None,
) -> bool:
    """
    Check epsilon handling for the value of a function that accepts a single value and an epsilon.

    For epsilon to be handled correctly, the function must evaluate to a non-singularity at
    x=singularity given epsilon

    Args:
        func: A callable func(x, epsilon) with a singularity to test
        singularity: The location of the singularity in func
        limit_direction: The side of the singularity to test, defaults to right side only
        display_func: Function to call to display an expression or a string
        expected_value: The expected value at the singularity, if not provided it will be
                        computed as the limit
    """

    # Converting between SymPy and SymEngine breaks substitution afterwards, so we require
    # running with SymPy.
    assert symforce.get_symbolic_api() == "sympy"

    # Create symbols
    x = sf.Symbol("x", real=True)
    epsilon = sf.Symbol("epsilon", positive=True)

    is_correct = True

    # Evaluate expression
    expr_eps = func(x, epsilon)
    expr_raw = expr_eps.subs(epsilon, 0)

    # Sub in zero
    expr_eps_at_x_zero = expr_eps.subs(x, singularity)
    if expr_eps_at_x_zero == sf.S.NaN:
        if display_func:
            display_func("Expressions (raw / eps):")
            display_func(expr_raw)
            display_func(expr_eps)

            display_func("[ERROR] Epsilon handling failed, expression at 0 is NaN.")
        is_correct = False

    # Take constant approximation at singularity and check equivalence
    if expected_value is None:
        value_x0_raw = _limit_and_simplify(expr_raw, x, singularity, limit_direction)
    else:
        value_x0_raw = expected_value
    value_x0_eps = expr_eps.subs(x, singularity)
    value_x0_eps_sub2 = _limit_and_simplify(value_x0_eps, epsilon, 0, "+")
    if value_x0_eps_sub2 != value_x0_raw:
        if display_func:
            # Only show the original expressions if we didn't show already
            if is_correct:
                display_func("Expressions (raw / eps):")
                display_func(expr_raw)
                display_func(expr_eps)

            display_func(f"[ERROR] Values at x={singularity} not match (raw / eps / eps.limit):")
            display_func(value_x0_raw)
            display_func(value_x0_eps)
            display_func(value_x0_eps_sub2)
        is_correct = False

    return is_correct


def is_derivative_with_epsilon_correct(
    func: T.Callable[[sf.Scalar, sf.Scalar], sf.Expr],
    singularity: sf.Scalar = 0,
    limit_direction: str = "+",
    display_func: T.Callable[[T.Any], None] = _default_display_func,
    expected_derivative: sf.Scalar = None,
) -> bool:
    """
    Check epsilon handling for the derivative of a function that accepts a single value and an
    epsilon.

    For epsilon to be handled correctly, a linear approximation of the original must match that
    taken with epsilon then substituted to zero

    Args:
        func: A callable func(x, epsilon) with a singularity to test
        singularity: The location of the singularity in func
        limit_direction: The side of the singularity to test, defaults to right side only
        display_func: Function to call to display an expression or a string
        expected_derivative: The expected derivative at the singularity, if not provided it will
                             be computed as the limit
    """

    # Converting between SymPy and SymEngine breaks substitution afterwards, so we require
    # running with SymPy.
    assert symforce.get_symbolic_api() == "sympy"

    # Create symbols
    x = sf.Symbol("x", real=True)
    epsilon = sf.Symbol("epsilon", positive=True)

    is_correct = True

    # Evaluate expression
    expr_eps = func(x, epsilon)
    expr_raw = expr_eps.subs(epsilon, 0)

    # Take linear approximation at singularity and check equivalence
    if expected_derivative is None:
        derivative_x0_raw = _limit_and_simplify(expr_raw.diff(x), x, singularity, limit_direction)
    else:
        derivative_x0_raw = expected_derivative
    derivative_x0_eps = expr_eps.diff(x).subs(x, singularity)
    derivative_x0_eps_sub2 = _limit_and_simplify(derivative_x0_eps, epsilon, 0, "+")
    if derivative_x0_eps_sub2 != derivative_x0_raw:
        if display_func:
            display_func("Expressions (raw / eps):")
            display_func(expr_raw)
            display_func(expr_eps)

            display_func(
                f"[ERROR] Derivatives at x={singularity} not match (raw / eps / eps.limit):"
            )
            display_func(derivative_x0_raw)
            display_func(derivative_x0_eps)
            display_func(derivative_x0_eps_sub2)
        is_correct = False

    return is_correct


def is_epsilon_correct(
    func: T.Callable[[sf.Scalar, sf.Scalar], sf.Scalar],
    singularity: sf.Scalar = 0,
    limit_direction: str = "+",
    display_func: T.Callable[[T.Any], None] = _default_display_func,
    expected_value: sf.Scalar = None,
    expected_derivative: sf.Scalar = None,
) -> bool:
    """
    Check epsilon handling for a function that accepts a single value and an epsilon.

    For epsilon to be handled correctly, the function must:
        1) evaluate to a non-singularity at x=singularity given epsilon
        2) linear approximation of the original must match that taken with epsilon then
           substituted to zero

    Args:
        func: A callable func(x, epsilon) with a singularity to test
        singularity: The location of the singularity in func
        limit_direction: The side of the singularity to test, defaults to right side only
        display_func: Function to call to display an expression or a string
        expected_value: The expected value at the singularity, if not provided it will be
                        computed as the limit
        expected_derivative: The expected derivative at the singularity, if not provided it will
                             be computed as the limit
    """
    is_value_correct = is_value_with_epsilon_correct(
        func, singularity, limit_direction, display_func, expected_value
    )

    is_derivative_correct = is_derivative_with_epsilon_correct(
        func, singularity, limit_direction, display_func, expected_derivative
    )

    return is_value_correct and is_derivative_correct
