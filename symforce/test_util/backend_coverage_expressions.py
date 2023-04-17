# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Utilities for generating expressions that provide good test coverage for new language backends
"""

import symforce.symbolic as sf
from symforce import typing as T


def backend_test_function(x: sf.Scalar, y: sf.Scalar) -> T.Tuple[sf.Scalar, ...]:
    """
    Given input symbols `x` and `y`, return a list of expressions which provide good test coverage
    over symbolic functions supported by symforce.

    The intention is that generating this function for a given backend should provide good test
    coverage indicating that the printer for that backend is implemented correctly.

    This does not attempt to test the rest of the backend (any geo, cam, matrix, or DataBuffer use),
    just the printer itself.
    """
    constants = (
        sf.S.Zero,
        sf.S.One,
        sf.S.Half,
        1.6,
        sf.Rational(11, 52),
        sf.S.Exp1,
        1 / sf.log(2),
        sf.log(2),
        sf.log(10),
        sf.S.Pi,
        sf.S.Pi / 2,
        sf.S.Pi / 4,
        1 / sf.S.Pi,
        2 / sf.S.Pi,
        2 / sf.sqrt(sf.S.Pi),
        sf.sqrt(2),
        1 / sf.sqrt(2),
    )

    unary_ops = (
        sf.Abs,
        sf.sin,
        sf.cos,
        sf.tan,
        sf.asin,
        sf.acos,
        sf.atan,
        sf.exp,
        sf.log,
        sf.sinh,
        sf.cosh,
        sf.tanh,
        sf.floor,
        sf.ceiling,
        sf.sqrt,
        sf.sympy.loggamma,
        sf.sympy.erfc,
        sf.sympy.asinh,
        sf.sympy.acosh,
        sf.sympy.atanh,
        sf.sympy.erf,
        sf.sympy.gamma,
        lambda x: sf.Mod(x, 5.5),
        lambda x: x + 1,
        lambda x: 2 * x,
        lambda x: x ** 2,
        lambda x: x ** 3,
        lambda x: x ** 4,
        lambda x: x ** 5,
        lambda x: x ** sf.S.Half,
        lambda x: x ** sf.Rational(3, 2),
        lambda x: sf.Max(0, x).diff(x),  # The heaviside function
    )

    binary_ops = (
        sf.sympy.atan2,
        sf.Max,
        sf.Min,
        sf.Mod,
        lambda x, y: x + y,
        lambda x, y: x * y,
        lambda x, y: x ** y,
        lambda x, y: (x + y) ** 2,
        lambda x, y: (x + y) ** 3,
    )

    return tuple(list(constants) + [op(x) for op in unary_ops] + [op(x, y) for op in binary_ops])
