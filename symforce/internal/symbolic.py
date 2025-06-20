# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

from __future__ import annotations

"""
The (symforce-internal) core symbolic API

This represents the core functions that comprise SymForce's unified version of the SymPy API,
without additional imports that would cause import cycles.  This means this module is safe to be
imported from those modules within SymForce.  Users should instead import :mod:`symforce.symbolic`,
which includes the entire SymForce symbolic API.

This combines functions available from various sources:

- Many functions we expose from the SymPy (and SymEngine) API
- Additional functions defined here to override those provided by SymPy or SymEngine, or provide a
  uniform interface between the two.  See https://symforce.org/api/symforce.symbolic.html for
  information on these
- Logic functions defined in :mod:`symforce.logic`, see the documentation for that module

It typically isn't necessary to actually access the symbolic API being used internally, but that is
available as well as :mod:`symforce.symbolic.sympy`.
"""

# ruff: noqa: F401, F403, F405, SLF001

import contextlib
import functools

import symforce
from symforce import logger
from symforce import typing as T

if symforce._symbolic_api is None:
    import textwrap

    logger.warning(
        textwrap.dedent(
            """
            No SYMFORCE_SYMBOLIC_API set, no symengine found, using sympy.

            For best performance during symbolic manipulation and code generation, use the symengine
            symbolic API, which should come installed with SymForce.  If you've installed SymForce
            without intentionally disabling SymEngine, and you are seeing this warning, this is a
            bug - please submit an issue: https://github.com/symforce-org/symforce/issues.

            To silence this warning, call `symforce.set_symbolic_api("sympy")` or set
            `SYMFORCE_SYMBOLIC_API=sympy` in your environment before importing `symforce.symbolic`.
            """
        )
    )

# See `symforce/__init__.py` for more information, this is used to check whether things that this
# module depends on are modified after importing
symforce._have_imported_symbolic = True

# Import sympy, for methods that convert to sympy types or call sympy functions below
import sympy as _sympy_py

if not T.TYPE_CHECKING and symforce.get_symbolic_api() == "symengine":
    sympy = symforce._find_symengine()

    # isort: split

    from symengine import Abs
    from symengine import Add
    from symengine import Basic
    from symengine import Catalan
    from symengine import Contains
    from symengine import Derivative
    from symengine import Dummy
    from symengine import E
    from symengine import EmptySet
    from symengine import EulerGamma
    from symengine import Expr
    from symengine import FiniteSet
    from symengine import Float
    from symengine import Function
    from symengine import GoldenRatio
    from symengine import I
    from symengine import Integer
    from symengine import Integers
    from symengine import Interval
    from symengine import KroneckerDelta
    from symengine import LambertW
    from symengine import LeviCivita
    from symengine import Max
    from symengine import Min
    from symengine import Mod
    from symengine import Mul
    from symengine import Number
    from symengine import Piecewise
    from symengine import Pow
    from symengine import Rational
    from symengine import RealNumber
    from symengine import Reals
    from symengine import S
    from symengine import Subs
    from symengine import SympifyError
    from symengine import UnevaluatedExpr
    from symengine import UniversalSet
    from symengine import acos
    from symengine import acosh
    from symengine import acot
    from symengine import acoth
    from symengine import acsc
    from symengine import acsch
    from symengine import asec
    from symengine import asech
    from symengine import asin
    from symengine import asinh
    from symengine import atan
    from symengine import atanh
    from symengine import beta
    from symengine import ceiling
    from symengine import conjugate
    from symengine import cos
    from symengine import cosh
    from symengine import cot
    from symengine import coth
    from symengine import csc
    from symengine import csch
    from symengine import cse
    from symengine import diff
    from symengine import digamma
    from symengine import dirichlet_eta
    from symengine import erf
    from symengine import erfc
    from symengine import exp
    from symengine import expand
    from symengine import floor
    from symengine import gamma
    from symengine import init_printing
    from symengine import integer_nthroot
    from symengine import isprime
    from symengine import lambdify
    from symengine import latex
    from symengine import linsolve
    from symengine import log
    from symengine import loggamma
    from symengine import lowergamma
    from symengine import nan
    from symengine import oo
    from symengine import perfect_power
    from symengine import pi
    from symengine import polygamma
    from symengine import sec
    from symengine import sech
    from symengine import series
    from symengine import sign
    from symengine import sin
    from symengine import sinh
    from symengine import sqrt
    from symengine import sqrt_mod
    from symengine import sympify
    from symengine import tan
    from symengine import tanh
    from symengine import trigamma
    from symengine import uppergamma
    from symengine import var
    from symengine import zeta
    from symengine import zoo
elif symforce.get_symbolic_api() == "sympy":
    import sympy

    sympy.init_printing()

    from sympy import Abs
    from sympy import Add
    from sympy import Basic
    from sympy import Catalan
    from sympy import Contains
    from sympy import Derivative
    from sympy import Dummy
    from sympy import E
    from sympy import EmptySet
    from sympy import EulerGamma
    from sympy import Expr
    from sympy import FiniteSet
    from sympy import Float
    from sympy import Function
    from sympy import GoldenRatio
    from sympy import I
    from sympy import Integer
    from sympy import Integers
    from sympy import Interval
    from sympy import KroneckerDelta
    from sympy import LambertW
    from sympy import LeviCivita
    from sympy import Max
    from sympy import Min
    from sympy import Mod
    from sympy import Mul
    from sympy import Number
    from sympy import Piecewise
    from sympy import Pow
    from sympy import Rational
    from sympy import RealNumber
    from sympy import Reals
    from sympy import S
    from sympy import Subs
    from sympy import SympifyError
    from sympy import UnevaluatedExpr
    from sympy import UniversalSet
    from sympy import acos
    from sympy import acosh
    from sympy import acot
    from sympy import acoth
    from sympy import acsc
    from sympy import acsch
    from sympy import asec
    from sympy import asech
    from sympy import asin
    from sympy import asinh
    from sympy import atan
    from sympy import atanh
    from sympy import beta
    from sympy import ceiling
    from sympy import conjugate
    from sympy import cos
    from sympy import cosh
    from sympy import cot
    from sympy import coth
    from sympy import csc
    from sympy import csch
    from sympy import cse
    from sympy import diff
    from sympy import digamma
    from sympy import dirichlet_eta
    from sympy import erf
    from sympy import erfc
    from sympy import exp
    from sympy import expand
    from sympy import floor
    from sympy import gamma
    from sympy import init_printing
    from sympy import integer_nthroot
    from sympy import isprime
    from sympy import lambdify
    from sympy import latex
    from sympy import linsolve
    from sympy import log
    from sympy import loggamma
    from sympy import lowergamma
    from sympy import nan
    from sympy import oo
    from sympy import perfect_power
    from sympy import pi
    from sympy import polygamma
    from sympy import sec
    from sympy import sech
    from sympy import series
    from sympy import sign
    from sympy import sin
    from sympy import sinh
    from sympy import sqrt
    from sympy import sqrt_mod
    from sympy import sympify
    from sympy import tan
    from sympy import tanh
    from sympy import trigamma
    from sympy import uppergamma
    from sympy import var
    from sympy import zeta
    from sympy import zoo
else:
    raise symforce.InvalidSymbolicApiError(symforce.get_symbolic_api())


# --------------------------------------------------------------------------------
# Create scopes
# --------------------------------------------------------------------------------


def create_named_scope(scopes_list: T.List[str]) -> T.Callable:
    """
    Return a context manager that adds to the given list of name scopes. This is used to
    add scopes to symbol names for namespacing.
    """

    @contextlib.contextmanager
    def named_scope(scope: str) -> T.Iterator[None]:
        scopes_list.append(scope)

        # The body of the with block is executed inside the yield, this ensures we release the
        # scope if something in the block throws
        try:
            yield None
        finally:
            scopes_list.pop()

    return named_scope


# Nested scopes created with `sf.scope`, initialized to empty (symbols created with no added scope)
__scopes__: T.List[str] = []


def set_scope(scope: str) -> None:
    global __scopes__  # noqa: PLW0603
    __scopes__ = scope.split(".") if scope else []


def get_scope() -> str:
    return ".".join(__scopes__)


scope = create_named_scope(__scopes__)

# --------------------------------------------------------------------------------
# Override Symbol and symbols
# --------------------------------------------------------------------------------

if not T.TYPE_CHECKING and sympy.__package__ == "symengine":

    class Symbol(sympy.Symbol):
        def __init__(
            self,
            name: str,
            commutative: bool = True,
            real: bool = True,
            positive: T.Optional[bool] = None,
        ) -> None:
            scoped_name = ".".join(__scopes__ + [name])
            # mypy doesn't understand that sympy.Symbol defines __new__, which takes these args
            super().__init__(scoped_name, commutative=commutative, real=real, positive=positive)  # type: ignore[call-arg]

            # TODO(hayk): This is not enabled, right now all symbols are commutative, real, but
            # not positive.
            # self.is_real = real
            # self.is_positive = positive
            # self.is_commutative = commutative

    sympy.Symbol = Symbol  # type: ignore[misc, assignment]

    # Because we're creating a new subclass, we also need to override sm.symbols to use this one
    _original_symbols = sympy.symbols

    @functools.wraps(_original_symbols)
    def symbols(names: str, **args: T.Any) -> T.Union[T.Sequence[Symbol], Symbol]:
        cls = args.pop("cls", Symbol)
        return _original_symbols(names, **dict(args, cls=cls))

    sympy.symbols = symbols

elif sympy.__package__ == "sympy":
    from sympy import Symbol
    from sympy import symbols

    # Save original
    _original_symbol_new = sympy.Symbol.__new__

    @staticmethod  # type: ignore[misc]  # staticmethod used with a non-method
    @functools.wraps(_original_symbol_new)
    def new_symbol(
        cls: T.Any,
        name: str,
        commutative: bool = True,
        real: bool = True,
        positive: T.Optional[bool] = None,
    ) -> None:
        name = ".".join(__scopes__ + [name])
        obj = _original_symbol_new(cls, name, commutative=commutative, real=real, positive=positive)
        return obj

    sympy.Symbol.__new__ = new_symbol  # type: ignore[assignment]
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)


# --------------------------------------------------------------------------------
# Default epsilon
# --------------------------------------------------------------------------------


from symforce import numeric_epsilon


def epsilon() -> T.Any:
    """
    The default epsilon for SymForce

    Library functions that require an epsilon argument should use a function signature like::

        def foo(x: Scalar, epsilon: Scalar = sf.epsilon()) -> Scalar:
            ...

    This makes it easy to configure entire expressions that make extensive use of epsilon to either
    use no epsilon (i.e. 0), or a symbol, or a numerical value.  It also means that by setting the
    default to a symbol, you can confidently generate code without worrying about having forgotten
    to pass an epsilon argument to one of these functions.

    For more information on how we use epsilon to prevent singularities, see the Epsilon Tutorial
    in the SymForce docs here: https://symforce.org/tutorials/epsilon_tutorial.html

    For purely numerical code that just needs a good default numerical epsilon, see
    :data:`symforce.symbolic.numeric_epsilon`.

    Returns:
        The current default epsilon.  This is typically some kind of "Scalar", like a float or a
        :class:`Symbol <symforce.symbolic.Symbol>`.
    """

    if isinstance(symforce._epsilon, symforce.SymbolicEpsilon):
        symforce._epsilon = sympy.Symbol(symforce._epsilon.name)

    symforce._have_used_epsilon = True

    return symforce._epsilon


# --------------------------------------------------------------------------------
# Typing
# --------------------------------------------------------------------------------

# isort: split
from symforce.typing import Scalar

# --------------------------------------------------------------------------------
# Logic functions
# --------------------------------------------------------------------------------

# isort: split
from symforce.logic import *

# --------------------------------------------------------------------------------
# Additional custom functions
# --------------------------------------------------------------------------------


def wrap_angle(x: Scalar) -> Scalar:
    """
    Wrap an angle to the interval [-pi, pi).  Commonly used to compute the shortest signed
    distance between two angles.

    See also: :func:`angle_diff`
    """
    return Mod(x + pi, 2 * pi) - pi


def angle_diff(x: Scalar, y: Scalar) -> Scalar:
    """
    Return the difference x - y, but wrapped into [-pi, pi); i.e. the angle `diff` closest to 0
    such that x = y + diff (mod 2pi).

    See also: :func:`wrap_angle`
    """
    return wrap_angle(x - y)


def argmax_onehot(vals: T.Iterable[Scalar], tolerance: Scalar = epsilon()) -> T.List[Scalar]:
    """
    Returns a list ``l`` such that ``l[i] = 1.0`` if ``i`` is the smallest index such that
    ``vals[i]`` equals ``Max(*vals)``. ``l[i] = 0.0`` otherwise.

    Precondition:
        vals has at least one element

    Args:
        tolerance: The selected max will be guaranteed to be within ``tolerance`` of the true max.
            If this is too small (e.g. if this is 0), the returned vector may be all zeros for some
            inputs.
    """
    vals = tuple(vals)
    m = Max(*vals) - tolerance
    results = []
    have_max_already = 0
    for val in vals:
        results.append(
            logical_and(
                is_nonnegative(val - m),
                logical_not(have_max_already, unsafe=True),
                unsafe=True,
            )
        )
        have_max_already = logical_or(have_max_already, results[-1], unsafe=True)
    return results


def argmax(vals: T.Iterable[Scalar]) -> Scalar:
    """
    Returns ``i`` (as a Scalar) such that ``i`` is the smallest index such that
    ``vals[i]`` equals ``Max(*vals)``.

    Precondition:
        vals has at least one element
    """
    return sum(i * val for i, val in enumerate(argmax_onehot(vals)))


def atan2(y: Scalar, x: Scalar, epsilon: Scalar = epsilon()) -> Scalar:
    return sympy.atan2(y, x + copysign_no_zero(epsilon, x))


def asin_safe(x: Scalar, epsilon: Scalar = epsilon()) -> Scalar:
    x_safe = Max(-1 + epsilon, Min(1 - epsilon, x))
    return sympy.asin(x_safe)


def acos_safe(x: Scalar, epsilon: Scalar = epsilon()) -> Scalar:
    x_safe = Max(-1 + epsilon, Min(1 - epsilon, x))
    return sympy.acos(x_safe)


def clamp(x: sf.Scalar, min_value: sf.Scalar, max_value: sf.Scalar) -> sf.Scalar:
    """
    Clamps x between min_value and max_value

    Returns::

        min_value if x < min_value
        x if min_value <= x <= max_value
        max_value if x > max_value

    Args:
        x: Value to clamp between min_value and max_value
        min_value: Scalar of same type and units as x; minimum value to return
        max_value: Scalar of same type and units as x; maximum value to return. Must be greater
            than min_value.
    """
    return sf.Min(max_value, sf.Max(min_value, x))


def set_eval_on_sympify(eval_on_sympy: bool = True) -> None:
    """
    When using the symengine backed, set whether we should eval args when converting objects to
    sympy.

    By default, this is enabled since this is the implicit behavior with stock symengine.
    Disabling eval results in more slightly ops, but greatly speeds up codegen time.
    """
    assert sympy is not None
    if sympy.__package__ == "symengine":
        import symengine.lib.symengine_wrapper as wrapper

        wrapper.__EVAL_ON_SYMPY__ = eval_on_sympy

    else:
        logger.debug("set_eval_on_sympify has no effect when not using symengine")


# --------------------------------------------------------------------------------
# Custom functions sign_no_zero and copysign_no_zero
# --------------------------------------------------------------------------------


class SignNoZero(_sympy_py.Function):
    @classmethod
    def eval(cls, *args: T.Any) -> T.Any:
        # NOTE(aaron): This doesn't implement nice simplifications like
        #
        #     x * sign_no_zero(y) -> copysign_no_zero(x, y) if x > 0
        #
        # We can't implement those on a Function afaik.  We should implement pre-CSE
        # optimizations that do this
        if len(args) != 1:
            raise ValueError(f"sign_no_zero takes one argument, got {len(args)}")

        x = args[0]
        if x.is_positive:
            return _sympy_py.S.One
        elif x.is_negative:
            return _sympy_py.S.NegativeOne
        elif x.is_zero:
            return _sympy_py.S.One
        else:
            return None

    # --------------------------------------------------------------------------------
    # Override print functions for SymPy printers
    #
    # For SymForce-only printers, like PyTorch, we don't need to override here.  We do this so that
    # printing these works if you use the actual SymPy printers (like
    # sympy.printing.pycode.PythonCodePrinter) for something, in addition to the SymForce subclasses
    # (like symforce.codegen.backends.python.python_code_printer.PythonCodePrinter)
    # --------------------------------------------------------------------------------

    def _sympystr(self, printer: _sympy_py.printing.StrPrinter, **kwargs: T.Any) -> T.Any:
        return f"sign_no_zero({printer._print(self.args[0], **kwargs)})"

    def _pretty(self, printer: T.Any, **kwargs: T.Any) -> T.Any:
        return printer._print_Function(self, func_name="sign_no_zero")

    def _latex(self, printer: T.Any, **kwargs: T.Any) -> T.Any:
        name = printer._hprint_Function("sign_no_zero")
        arg = printer._print(self.args[0], **kwargs)
        can_fold_brackets = printer._settings[
            "fold_func_brackets"
        ] and not printer._needs_function_brackets(self.args[0])
        if can_fold_brackets:
            return f"{name}{arg}"
        else:
            return rf"{name}{{\left({arg} \right)}}"

    def _pythoncode(self, printer: T.Any, **kwargs: T.Any) -> T.Any:
        f = printer._module_format("math.copysign")
        arg0 = printer._print(self.args[0], **kwargs)
        return f"{f}(1, {arg0})"

    def _eval_evalf(self, prec: T.Any) -> T.Any:
        return (2 * _sympy_py.Min(_sympy_py.sign(self.args[0]), 0) + 1)._eval_evalf(prec)

    @staticmethod
    def fdiff(argindex: T.Optional[int] = 1) -> T.Any:
        assert argindex == 1
        return _sympy_py.S.Zero


SymPySignNoZero = SignNoZero


class CopysignNoZero(_sympy_py.Function):
    @classmethod
    def eval(cls, *args: T.Any) -> T.Any:
        if len(args) != 2:
            raise ValueError(f"copysign_no_zero takes two arguments, got {len(args)}")

        x = args[0]
        y = args[1]

        if x.is_zero:
            return x
        elif x.is_positive and (y.is_positive or y.is_zero):
            return x
        elif x.is_negative and y.is_negative:
            return x
        elif x.is_negative and (y.is_positive or y.is_zero):
            return -x
        elif x.is_positive and y.is_negative:
            return -x
        else:
            return None

    # --------------------------------------------------------------------------------
    # Override print functions for SymPy printers
    #
    # For SymForce-only printers, like PyTorch, we don't need to override here.  We do this so that
    # printing these works if you use the actual SymPy printers (like
    # sympy.printing.pycode.PythonCodePrinter) for something, in addition to the SymForce subclasses
    # (like symforce.codegen.backends.python.python_code_printer.PythonCodePrinter)
    # --------------------------------------------------------------------------------

    def _sympystr(self, printer: _sympy_py.printing.StrPrinter, **kwargs: T.Any) -> T.Any:
        arg0 = printer._print(self.args[0], **kwargs)
        arg1 = printer._print(self.args[1], **kwargs)
        return f"copysign_no_zero({arg0}, {arg1})"

    def _pretty(self, printer: T.Any, **kwargs: T.Any) -> T.Any:
        return printer._print_Function(self, func_name="copysign_no_zero")

    def _latex(self, printer: T.Any, **kwargs: T.Any) -> T.Any:
        name = printer._hprint_Function("copysign_no_zero")
        arg0 = printer._print(self.args[0], **kwargs)
        arg1 = printer._print(self.args[1], **kwargs)
        return rf"{name}{{\left({arg0}, {arg1} \right)}}"

    def _pythoncode(self, printer: T.Any, **kwargs: T.Any) -> T.Any:
        f = printer._module_format("math.copysign")
        arg0 = printer._print(self.args[0], **kwargs)
        arg1 = printer._print(self.args[1], **kwargs)
        return f"{f}({arg0}, {arg1})"

    def _eval_evalf(self, prec: T.Any) -> T.Any:
        return (_sympy_py.Abs(self.args[0]) * SymPySignNoZero(self.args[1]))._eval_evalf(prec)

    def fdiff(self, argindex: T.Optional[int] = 1) -> T.Any:
        if argindex == 1:
            return SymPySignNoZero(self.args[0]) * SymPySignNoZero(self.args[1])
        elif argindex == 2:
            return _sympy_py.S.Zero


SymPyCopysignNoZero = CopysignNoZero


if not T.TYPE_CHECKING and sympy.__package__ == "symengine":
    del SignNoZero
    del CopysignNoZero
    from symengine import CopysignNoZero
    from symengine import SignNoZero
elif sympy.__package__ == "sympy":
    pass
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)


def sign_no_zero(x: Scalar) -> sf.Expr:
    """
    Returns -1 if x is negative, 1 if x is positive, and 1 if x is zero.
    """
    return SignNoZero(x)


def copysign_no_zero(x: Scalar, y: Scalar) -> Scalar:
    """
    Returns a value with the magnitude of x and sign of y

    If y is zero, returns positive x.
    """
    return CopysignNoZero(x, y)


# --------------------------------------------------------------------------------
# Create simplify
# --------------------------------------------------------------------------------

if not T.TYPE_CHECKING and sympy.__package__ == "symengine":

    def simplify(*args: T.Any, **kwargs: T.Any) -> Scalar:
        logger.warning("Converting to sympy to use .simplify")
        return sympy.S(_sympy_py.simplify(_sympy_py.S(*args), **kwargs))

elif sympy.__package__ == "sympy":
    from sympy.simplify import simplify
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)


# --------------------------------------------------------------------------------
# Create limit
# --------------------------------------------------------------------------------

if not T.TYPE_CHECKING and sympy.__package__ == "symengine":

    def limit(
        e: T.Any,
        z: T.Any,
        z0: T.Any,
        dir: str = "+",  # noqa: A002
    ) -> Scalar:
        logger.warning("Converting to sympy to use .limit")
        return sympy.S(_sympy_py.limit(_sympy_py.S(e), _sympy_py.S(z), _sympy_py.S(z0), dir=dir))

elif sympy.__package__ == "sympy":
    from sympy import limit
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)

# --------------------------------------------------------------------------------
# Create integrate
# --------------------------------------------------------------------------------

if not T.TYPE_CHECKING and sympy.__package__ == "symengine":

    def integrate(
        *args: T.Any,
        meijerg: T.Optional[bool] = None,
        conds: T.Literal["piecewise", "separate", "none"] = "piecewise",
        risch: T.Optional[bool] = None,
        heurisch: T.Optional[T.Any] = None,
        manual: T.Optional[bool] = None,
        **kwargs: T.Any,
    ):
        logger.warning("Converting to sympy to use .integrate")
        return sympy.S(
            _sympy_py.integrate(
                *[_sympy_py.S(arg) for arg in args],
                meijerg=meijerg,
                conds=conds,
                risch=risch,
                heurisch=heurisch,
                manual=manual,
                **kwargs,
            )
        )

elif sympy.__package__ == "sympy":
    from sympy import integrate
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)

# --------------------------------------------------------------------------------
# Override solve
# --------------------------------------------------------------------------------

if not T.TYPE_CHECKING and sympy.__package__ == "symengine":
    # Unfortunately this already doesn't have a docstring or argument list in symengine
    def solve(*args: T.Any, **kwargs: T.Any) -> T.List[Scalar]:
        solution = sympy.solve(*args, **kwargs)
        from symengine.lib.symengine_wrapper import EmptySet as _EmptySet

        if isinstance(solution, FiniteSet):
            return list(solution.args)
        elif isinstance(solution, _EmptySet):
            return []
        else:
            raise NotImplementedError(
                f"sm.solve currently only supports FiniteSet and EmptySet results on SymEngine, got {type(solution)} instead"
            )

elif sympy.__package__ == "sympy":
    from sympy import solve
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)

# --------------------------------------------------------------------------------
# Override count_ops
# --------------------------------------------------------------------------------

if not T.TYPE_CHECKING and sympy.__package__ == "symengine":
    from symengine import count_ops
elif sympy.__package__ == "sympy":
    from symforce._sympy_count_ops import count_ops
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)

# --------------------------------------------------------------------------------
# Add DataBuffer
# --------------------------------------------------------------------------------

if sympy.__package__ == "symengine":
    from symengine import DataBuffer
elif sympy.__package__ == "sympy":
    from symforce.databuffer import DataBuffer

    # TODO(aaron): Is this still necessary?
    DataBuffer.__sympy_module__ = sympy
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)

# --------------------------------------------------------------------------------
# Add derivatives
# --------------------------------------------------------------------------------

if sympy.__package__ == "symengine":
    pass
elif sympy.__package__ == "sympy":
    # Hack in some key derivatives that sympy doesn't do. For all these cases the derivatives
    # here are correct except at the discrete switching point, which is correct for our
    # numerical purposes.
    floor._eval_derivative = lambda s, v: S.Zero
    sign._eval_derivative = lambda s, v: S.Zero

    def mod_derivative(self: T.Any, x: T.Any) -> T.Any:
        p, q = self.args
        return self._eval_rewrite_as_floor(p, q).diff(x)

    Mod._eval_derivative = mod_derivative
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)

# --------------------------------------------------------------------------------
# Create more powerful subs
# --------------------------------------------------------------------------------


def _flatten_storage_type_subs(
    subs_pairs: T.Sequence[T.Tuple[T.Any, T.Any]],
) -> T.Dict[T.Any, T.Any]:
    """
    Replace storage types with their scalar counterparts
    """
    new_subs_dict = {}
    # Import these lazily, since initialization.py is imported from symforce/__init__.py
    from symforce import ops
    from symforce import typing_util

    for key, value in subs_pairs:
        if key is None:
            # Sympy seems to support None keys and do nothing with them
            continue

        if typing_util.scalar_like(key):
            assert typing_util.scalar_like(value)
            new_subs_dict[key] = value
            continue

        if isinstance(key, DataBuffer) or isinstance(value, DataBuffer):
            assert isinstance(value, type(key)) or isinstance(key, type(value))
            new_subs_dict[key] = value
            continue

        try:
            new_keys = ops.StorageOps.to_storage(key)
            new_values = ops.StorageOps.to_storage(value)
        except NotImplementedError:
            new_subs_dict[key] = value
        else:
            error_msg = f"value type {type(value)} is not an instance of key type {type(key)}"
            assert isinstance(value, type(key)) or isinstance(key, type(value)), error_msg
            if len(new_keys) != len(new_values):
                raise ValueError(
                    f"number of keys ({len(new_keys)}) do not match number of values ({len(new_values)})"
                )
            for new_key, new_value in zip(new_keys, new_values):
                new_subs_dict[new_key] = new_value
    return new_subs_dict


def _get_subs_dict(*args: T.Any, dont_flatten_args: bool = False, **kwargs: T.Any) -> T.Dict:
    """
    Handle args to subs being a single key-value pair or a dict.

    Keyword Args:
        dont_flatten_args: if true and args is a single argument, assume that args is a dict mapping
            scalar expressions to other scalar expressions. i.e. StorageOps flattening will **not**
            occur. This is significantly faster.

        **kwargs is unused but needed for sympy compatibility
    """
    if len(args) == 2:
        subs_pairs = [(args[0], args[1])]
    elif len(args) == 1:
        if dont_flatten_args:
            assert isinstance(args[0], T.Dict)
            return args[0]
        if isinstance(args[0], T.Mapping):
            subs_pairs = list(args[0].items())
        else:
            subs_pairs = args[0]
    else:
        raise ValueError(
            "subs must be passed either two positional arguments (a key-value pair), or a dict of "
            f"key-value pairs.  Instead, got {args}"
        )

    assert isinstance(subs_pairs, T.Sequence)
    return _flatten_storage_type_subs(subs_pairs)


if sympy.__package__ == "symengine":
    # For some reason this doesn't exist unless we import the symengine_wrapper directly as a
    # local variable, i.e. just `import symengine.lib.symengine_wrapper` does not let us access
    # symengine.lib.symengine_wrapper
    import symengine.lib.symengine_wrapper as wrapper

    original_get_dict = wrapper.get_dict
    wrapper.get_dict = lambda *args, **kwargs: original_get_dict(_get_subs_dict(*args, **kwargs))
elif sympy.__package__ == "sympy":
    original_subs = sympy.Basic.subs
    sympy.Basic.subs = lambda self, *args, **kwargs: original_subs(  # type: ignore[method-assign]
        self, _get_subs_dict(*args, **kwargs), **kwargs
    )
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)
