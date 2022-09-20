# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
This file defines a modified version of sympy.count_ops which is more suited
to our needs.

Based on: https://github.com/sympy/sympy/blob/c1deac1ce6c989ee2c66fc505595603cca77681f/sympy/core/function.py#L3059
with light modification.

Note, type checking is disabled on the functions in this module with the decorator
@T.no_type_check. Fundamentally, this is because the code relies heavily on
object's fields to deduce their subtype (which mypy isn't able to understand).
We're not interested in modifying the code to pass the linter because we wanted
to change as little as possible from the original sympy code (to better distinguish
our own material changes).

Currently, the body of of count_ops is the body as found in sympy, modified only
with those comments, formatting, and material changes to the implementation.
"""

from sympy import Add
from sympy import Basic
from sympy import Derivative
from sympy import Expr
from sympy import S
from sympy import Symbol
from sympy import sympify
from sympy.core.function import UndefinedFunction
from sympy.core.operations import LatticeOp
from sympy.utilities.iterables import iterable

from symforce import typing as T
from symforce.typing import Dict


@T.no_type_check
def _coeff_isneg(a: Basic) -> bool:
    """Return True if the leading Number is negative.
    Examples
    ========
    >>> from sympy.core.function import _coeff_isneg
    >>> from sympy import S, Symbol, oo, pi
    >>> _coeff_isneg(-3*pi)
    True
    >>> _coeff_isneg(S(3))
    False
    >>> _coeff_isneg(-oo)
    True
    >>> _coeff_isneg(Symbol('n', negative=True)) # coeff is 1
    False
    For matrix expressions:
    >>> from sympy import MatrixSymbol, sqrt
    >>> A = MatrixSymbol("A", 3, 3)
    >>> _coeff_isneg(-sqrt(2)*A)
    True
    >>> _coeff_isneg(sqrt(2)*A)
    False
    """

    if a.is_MatMul:
        a = a.args[0]
    if a.is_Mul:
        a = a.args[0]
    return a.is_Number and a.is_extended_negative


@T.no_type_check
def count_ops(expr: T.Any, visual: bool = False) -> T.Union[Expr, int]:
    """
    Return a representation (integer or expression) of the operations in expr.
    Parameters
    ==========
    expr : Expr
        If expr is an iterable, the sum of the op counts of the
        items will be returned.
    visual : bool, optional
        If ``False`` (default) then the sum of the coefficients of the
        visual expression will be returned.
        If ``True`` then the number of each type of operation is shown
        with the core class types (or their virtual equivalent) multiplied by the
        number of times they occur.
    Examples
    ========
    >>> from sympy.abc import a, b, x, y
    >>> from sympy import sin, count_ops
    Although there isn't a SUB object, minus signs are interpreted as
    either negations or subtractions:
    >>> (x - y).count_ops(visual=True)
    SUB
    >>> (-x).count_ops(visual=True)
    NEG
    Here, there are two Adds and a Pow:
    >>> (1 + a + b**2).count_ops(visual=True)
    2*ADD + POW
    In the following, an Add, Mul, Pow and two functions:
    >>> (sin(x)*x + sin(x)**2).count_ops(visual=True)
    ADD + MUL + POW + 2*SIN
    for a total of 5:
    >>> (sin(x)*x + sin(x)**2).count_ops(visual=False)
    5
    Note that "what you type" is not always what you get. The expression
    1/x/y is translated by sympy into 1/(x*y) so it gives a DIV and MUL rather
    than two DIVs:
    >>> (1/x/y).count_ops(visual=True)
    DIV + MUL
    The visual option can be used to demonstrate the difference in
    operations for expressions in different forms. Here, the Horner
    representation is compared with the expanded form of a polynomial:
    >>> eq=x*(1 + x*(2 + x*(3 + x)))
    >>> count_ops(eq.expand(), visual=True) - count_ops(eq, visual=True)
    -MUL + 3*POW
    The count_ops function also handles iterables:
    >>> count_ops([x, sin(x), None, True, x + 2], visual=False)
    2
    >>> count_ops([x, sin(x), None, True, x + 2], visual=True)
    ADD + SIN
    >>> count_ops({x: sin(x), x + 2: y + 1}, visual=True)
    2*ADD + SIN
    """
    from sympy import Integral
    from sympy import Sum
    from sympy.core.relational import Relational
    from sympy.logic.boolalg import BooleanFunction
    from sympy.simplify.radsimp import fraction
    from sympy.utilities.misc import func_name

    expr = sympify(expr)

    # pylint: disable=too-many-nested-blocks
    if isinstance(expr, Expr) and not expr.is_Relational:

        ops = []
        args = [expr]
        NEG = Symbol("NEG")
        DIV = Symbol("DIV")
        SUB = Symbol("SUB")
        ADD = Symbol("ADD")
        EXP = Symbol("EXP")
        while args:
            a = args.pop()

            # if the following fails because the object is
            # not Basic type, then the object should be fixed
            # since it is the intention that all args of Basic
            # should themselves be Basic
            if a.is_Mul or a.is_MatMul:
                if _coeff_isneg(a):
                    ops.append(NEG)
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a
                n, d = fraction(a)
                if n.is_Integer:
                    ops.append(DIV)
                    if n < 0:
                        ops.append(NEG)
                    args.append(d)
                    continue  # won't be -Mul but could be Add
                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                    ops.append(DIV)
                    args.append(n)
                    continue  # could be -Mul
            elif a.is_Add or a.is_MatAdd:
                aargs = list(a.args)
                negs = 0
                for i, ai in enumerate(aargs):
                    if _coeff_isneg(ai):
                        negs += 1
                        args.append(-ai)
                        if i > 0:
                            ops.append(SUB)
                    else:
                        args.append(ai)
                        if i > 0:
                            ops.append(ADD)
                if negs == len(aargs):  # -x - y = NEG + SUB
                    ops.append(NEG)
                elif _coeff_isneg(aargs[0]):  # -x + y = SUB, but already recorded ADD
                    ops.append(SUB - ADD)
                continue
            if a.is_Pow and a.exp is S.NegativeOne:
                ops.append(DIV)
                args.append(a.base)  # won't be -Mul but could be Add
                continue
            if a == S.Exp1:
                ops.append(EXP)
                continue
            if a.is_Pow and a.base == S.Exp1:
                ops.append(EXP)
                args.append(a.exp)
                continue
            if a.is_Mul or isinstance(a, LatticeOp):
                o = Symbol(a.func.__name__.upper())
                # count the args
                ops.append(o * (len(a.args) - 1))
            elif a.args and (  # pylint: disable=too-many-boolean-expressions
                a.is_Pow  # pylint: disable=consider-merging-isinstance
                or a.is_Function
                or isinstance(a, Derivative)
                or isinstance(a, Integral)
                or isinstance(a, Sum)
            ):
                # if it's not in the list above we don't
                # consider a.func something to count, e.g.
                # Tuple, MatrixSymbol, etc...
                if isinstance(a.func, UndefinedFunction):
                    o = Symbol("FUNC_" + a.func.__name__.upper())
                else:
                    o = Symbol(a.func.__name__.upper())
                ops.append(o)

            if not a.is_Symbol:
                args.extend(a.args)

    elif isinstance(expr, Dict):
        ops = [count_ops(k, visual=visual) + count_ops(v, visual=visual) for k, v in expr.items()]
    elif iterable(expr):
        ops = [count_ops(i, visual=visual) for i in expr]
    elif isinstance(expr, (Relational, BooleanFunction)):
        ops = []
        for arg in expr.args:
            ops.append(count_ops(arg, visual=True))
        o = Symbol(func_name(expr, short=True).upper())
        ops.append(o)
    elif not isinstance(expr, Basic):
        ops = []
    else:  # it's Basic not isinstance(expr, Expr):
        if not isinstance(expr, Basic):
            raise TypeError("Invalid type of expr")
        else:
            ops = []
            args = [expr]
            while args:
                a = args.pop()

                if a.args:
                    o = Symbol(type(a).__name__.upper())
                    if a.is_Boolean:
                        ops.append(o * (len(a.args) - 1))
                    else:
                        ops.append(o)
                    args.extend(a.args)

    if not ops:
        if visual:
            return S.Zero
        return 0

    ops = Add(*ops)

    if visual:
        return ops

    if ops.is_Number:
        return int(ops)

    return sum(int((a.args or [1])[0]) for a in Add.make_args(ops))
