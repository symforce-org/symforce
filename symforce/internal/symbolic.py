# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
The (symforce-internal) core symbolic API

This represents the core functions that comprise SymForce's unified version of the SymPy API,
without additional imports that would cause import cycles.  This means this module is safe to be
imported from those modules within SymForce.  Users should instead import `symforce.symbolic`, which
includes the entire SymForce symbolic API.

This combines functions available from various sources:

- Many functions we expose from the SymPy (and SymEngine) API
- Additional functions defined here to override those provided by SymPy or SymEngine, or provide a
  uniform interface between the two.  See https://symforce.org/api/symforce.symbolic.html for
  information on these
- Logic functions defined in `symforce.logic`, see the documentation for that module

It typically isn't necessary to actually access the symbolic API being used internally, but that is
available as well as `symforce.symbolic.sympy`.
"""

# pylint: disable=unused-import
# pylint: disable=unused-wildcard-import
# pylint: disable=ungrouped-imports

import contextlib
import symforce
from symforce import typing as T
from symforce import logger

# See `symforce/__init__.py` for more information, this is used to check whether things that this
# module depends on are modified after importing
symforce._have_imported_symbolic = True  # pylint: disable=protected-access

if not T.TYPE_CHECKING and symforce.get_symbolic_api() == "symengine":
    sympy = symforce._find_symengine()  # pylint: disable=protected-access

    from symengine import (
        Abs,
        Add,
        Basic,
        Catalan,
        Contains,
        Derivative,
        Dummy,
        E,
        EmptySet,
        EulerGamma,
        Expr,
        FiniteSet,
        Float,
        Function,
        GoldenRatio,
        I,
        Integer,
        Integers,
        Interval,
        KroneckerDelta,
        LambertW,
        LeviCivita,
        Max,
        Min,
        Mod,
        Mul,
        Number,
        Piecewise,
        Pow,
        Rational,
        RealNumber,
        Reals,
        S,
        Subs,
        SympifyError,
        UnevaluatedExpr,
        UniversalSet,
        acos,
        acosh,
        acot,
        acoth,
        acsc,
        acsch,
        asec,
        asech,
        asin,
        asinh,
        atan,
        atanh,
        beta,
        ceiling,
        conjugate,
        cos,
        cosh,
        cot,
        coth,
        csc,
        csch,
        cse,
        diff,
        digamma,
        dirichlet_eta,
        erf,
        erfc,
        exp,
        expand,
        floor,
        gamma,
        init_printing,
        integer_nthroot,
        isprime,
        lambdify,
        latex,
        linsolve,
        log,
        loggamma,
        lowergamma,
        nan,
        oo,
        perfect_power,
        pi,
        polygamma,
        sec,
        sech,
        series,
        sign,
        sin,
        sinh,
        sqrt,
        sqrt_mod,
        sympify,
        tan,
        tanh,
        trigamma,
        uppergamma,
        var,
        zeta,
        zoo,
    )
elif symforce.get_symbolic_api() == "sympy":
    import sympy

    sympy.init_printing()

    from sympy import (
        Abs,
        Add,
        Basic,
        Catalan,
        Contains,
        Derivative,
        Dummy,
        E,
        EmptySet,
        EulerGamma,
        Expr,
        FiniteSet,
        Float,
        Function,
        GoldenRatio,
        I,
        Integer,
        Integers,
        Interval,
        KroneckerDelta,
        LambertW,
        LeviCivita,
        Max,
        Min,
        Mod,
        Mul,
        Number,
        Piecewise,
        Pow,
        Rational,
        RealNumber,
        Reals,
        S,
        Subs,
        SympifyError,
        UnevaluatedExpr,
        UniversalSet,
        acos,
        acosh,
        acot,
        acoth,
        acsc,
        acsch,
        asec,
        asech,
        asin,
        asinh,
        atan,
        atanh,
        beta,
        ceiling,
        conjugate,
        cos,
        cosh,
        cot,
        coth,
        csc,
        csch,
        cse,
        diff,
        digamma,
        dirichlet_eta,
        erf,
        erfc,
        exp,
        expand,
        floor,
        gamma,
        init_printing,
        integer_nthroot,
        isprime,
        lambdify,
        latex,
        linsolve,
        log,
        loggamma,
        lowergamma,
        nan,
        oo,
        perfect_power,
        pi,
        polygamma,
        sec,
        sech,
        series,
        sign,
        sin,
        sinh,
        sqrt,
        sqrt_mod,
        sympify,
        tan,
        tanh,
        trigamma,
        uppergamma,
        var,
        zeta,
        zoo,
    )
else:
    raise symforce.InvalidSymbolicApiError(symforce.get_symbolic_api())


# --------------------------------------------------------------------------------
# Default epsilon
# --------------------------------------------------------------------------------


from symforce import numeric_epsilon


def epsilon() -> T.Any:
    """
    The default epsilon for SymForce

    Library functions that require an epsilon argument should use a function signature like:

        def foo(x: Scalar, epsilon: Scalar = sf.epsilon()) -> Scalar:
            ...

    This makes it easy to configure entire expressions that make extensive use of epsilon to either
    use no epsilon (i.e. 0), or a symbol, or a numerical value.  It also means that by setting the
    default to a symbol, you can confidently generate code without worrying about having forgotten
    to pass an epsilon argument to one of these functions.

    For more information on how we use epsilon to prevent singularities, see the Epsilon Tutorial
    in the SymForce docs here: https://symforce.org/tutorials/epsilon_tutorial.html

    For purely numerical code that just needs a good default numerical epsilon, see
    `symforce.symbolic.numerical_epsilon`.

    Returns: The current default epsilon.  This is typically some kind of "Scalar", like a float or
             a Symbol.
    """
    symforce._have_used_epsilon = True  # pylint: disable=protected-access

    return symforce._epsilon  # pylint: disable=protected-access


# --------------------------------------------------------------------------------
# Override Symbol and symbols
# --------------------------------------------------------------------------------

if not T.TYPE_CHECKING and sympy.__package__ == "symengine":

    class Symbol(sympy.Symbol):  # pylint: disable=function-redefined,too-many-ancestors
        def __init__(
            self, name: str, commutative: bool = True, real: bool = True, positive: bool = None
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
    original_symbols = sympy.symbols

    def symbols(  # pylint: disable=function-redefined
        names: str, **args: T.Any
    ) -> T.Union[T.Sequence[Symbol], Symbol]:
        cls = args.pop("cls", Symbol)
        return original_symbols(names, **dict(args, cls=cls))

    sympy.symbols = symbols

elif sympy.__package__ == "sympy":
    from sympy import Symbol
    from sympy import symbols  # type: ignore[misc] # mypy is mad that this is of a different type on sympy

    # Save original
    original_symbol_new = sympy.Symbol.__new__

    @staticmethod  # type: ignore[misc]
    def new_symbol(
        cls: T.Any,
        name: str,
        commutative: bool = True,
        real: bool = True,
        positive: bool = None,
    ) -> None:
        name = ".".join(__scopes__ + [name])
        obj = original_symbol_new(cls, name, commutative=commutative, real=real, positive=positive)
        return obj

    sympy.Symbol.__new__ = new_symbol  # type: ignore[assignment]
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)


# --------------------------------------------------------------------------------
# Typing
# --------------------------------------------------------------------------------

from symforce.typing import Scalar


# --------------------------------------------------------------------------------
# Logic functions
# --------------------------------------------------------------------------------

from symforce.logic import *  # pylint: disable=wildcard-import


# --------------------------------------------------------------------------------
# Additional custom functions
# --------------------------------------------------------------------------------


def wrap_angle(x: Scalar) -> Scalar:
    """
    Wrap an angle to the interval [-pi, pi).  Commonly used to compute the shortest signed
    distance between two angles.

    See also: `angle_diff`
    """
    return Mod(x + pi, 2 * pi) - pi


def angle_diff(x: Scalar, y: Scalar) -> Scalar:
    """
    Return the difference x - y, but wrapped into [-pi, pi); i.e. the angle `diff` closest to 0
    such that x = y + diff (mod 2pi).

    See also: `wrap_angle`
    """
    return wrap_angle(x - y)


def sign_no_zero(x: Scalar) -> Scalar:
    """
    Returns -1 if x is negative, 1 if x is positive, and 1 if x is zero.
    """
    return 2 * Min(sign(x), 0) + 1


def copysign_no_zero(x: Scalar, y: Scalar) -> Scalar:
    """
    Returns a value with the magnitude of x and sign of y. If y is zero, returns positive x.
    """
    return Abs(x) * sign_no_zero(y)


def argmax_onehot(vals: T.Iterable[Scalar]) -> T.List[Scalar]:
    """
    Returns a list l such that l[i] = 1.0 if i is the smallest index such that
    vals[i] equals Max(*vals). l[i] = 0.0 otherwise.

    Precondition:
        vals has at least one element
    """
    vals = tuple(vals)
    m = Max(*vals)
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
    Returns i (as a Scalar) such that i is the smallest index such that
    vals[i] equals Max(*vals).

    Precondition:
        vals has at least one element
    """
    return sum(i * val for i, val in enumerate(argmax_onehot(vals)))


def atan2(y: Scalar, x: Scalar, epsilon: Scalar = epsilon()) -> Scalar:
    return sympy.atan2(y, x + (sign(x) + 0.5) * epsilon)


def asin_safe(x: Scalar, epsilon: Scalar = epsilon()) -> Scalar:
    x_safe = Max(-1 + epsilon, Min(1 - epsilon, x))
    return sympy.asin(x_safe)


def acos_safe(x: Scalar, epsilon: Scalar = epsilon()) -> Scalar:
    x_safe = Max(-1 + epsilon, Min(1 - epsilon, x))
    return sympy.acos(x_safe)


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
# Create simplify
# --------------------------------------------------------------------------------

if sympy.__package__ == "symengine":
    import sympy as _sympy_py

    def simplify(*args: T.Any, **kwargs: T.Any) -> Scalar:
        logger.warning("Converting to sympy to use .simplify")
        return sympy.S(_sympy_py.simplify(_sympy_py.S(*args), **kwargs))

elif sympy.__package__ == "sympy":
    from sympy import simplify  # type: ignore[misc] # incompatible import
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)


# --------------------------------------------------------------------------------
# Create limit
# --------------------------------------------------------------------------------

if sympy.__package__ == "symengine":
    import sympy as _sympy_py

    def limit(
        e: T.Any, z: T.Any, z0: T.Any, dir: str = "+"  # pylint: disable=redefined-builtin
    ) -> Scalar:
        logger.warning("Converting to sympy to use .limit")
        return sympy.S(_sympy_py.limit(_sympy_py.S(e), _sympy_py.S(z), _sympy_py.S(z0), dir=dir))

elif sympy.__package__ == "sympy":
    from sympy import limit  # type: ignore[misc] # incompatible import
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)

# --------------------------------------------------------------------------------
# Override solve
# --------------------------------------------------------------------------------

if sympy.__package__ == "symengine":
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
    from sympy import solve  # type: ignore[misc] # incompatible import
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)

# --------------------------------------------------------------------------------
# Override count_ops
# --------------------------------------------------------------------------------

if sympy.__package__ == "symengine":
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
    setattr(floor, "_eval_derivative", lambda s, v: S.Zero)
    setattr(sign, "_eval_derivative", lambda s, v: S.Zero)

    def mod_derivative(self: T.Any, x: T.Any) -> T.Any:
        p, q = self.args
        return self._eval_rewrite_as_floor(p, q).diff(x)  # pylint: disable=protected-access

    setattr(Mod, "_eval_derivative", mod_derivative)
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)

# --------------------------------------------------------------------------------
# Create more powerful subs
# --------------------------------------------------------------------------------


def _flatten_storage_type_subs(
    subs_pairs: T.Sequence[T.Tuple[T.Any, T.Any]]
) -> T.Dict[T.Any, T.Any]:
    """
    Replace storage types with their scalar counterparts
    """
    new_subs_dict = {}
    # Import these lazily, since initialization.py is imported from symforce/__init__.py
    from symforce import ops  # pylint: disable=cyclic-import
    from symforce import python_util  # pylint: disable=cyclic-import

    for key, value in subs_pairs:

        if python_util.scalar_like(key):
            assert python_util.scalar_like(value)
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
            for new_key, new_value in zip(new_keys, new_values):
                new_subs_dict[new_key] = new_value
    return new_subs_dict


def _get_subs_dict(*args: T.Any, dont_flatten_args: bool = False, **kwargs: T.Any) -> T.Dict:
    """
    Handle args to subs being a single key-value pair or a dict.

    Keyword Args:
        dont_flatten_args (bool): if true and args is a single argument, assume that args is a
            dict mapping scalar expressions to other scalar expressions. i.e. StorageOps flattening
            will *not* occur. This is significantly faster.

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

    assert isinstance(subs_pairs, T.Sequence)
    return _flatten_storage_type_subs(subs_pairs)


if sympy.__package__ == "symengine":
    # For some reason this doesn't exist unless we import the symengine_wrapper directly as a
    # local variable, i.e. just `import symengine.lib.symengine_wrapper` does not let us access
    # symengine.lib.symengine_wrapper
    import symengine.lib.symengine_wrapper as wrapper  # pylint: disable=no-name-in-module

    original_get_dict = wrapper.get_dict
    wrapper.get_dict = lambda *args, **kwargs: original_get_dict(_get_subs_dict(*args, **kwargs))
elif sympy.__package__ == "sympy":
    original_subs = sympy.Basic.subs
    sympy.Basic.subs = lambda self, *args, **kwargs: original_subs(  # type: ignore[assignment]
        self, _get_subs_dict(*args, **kwargs), **kwargs
    )
else:
    raise symforce.InvalidSymbolicApiError(sympy.__package__)

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
__scopes__ = []


def set_scope(scope: str) -> None:
    global __scopes__  # pylint: disable=global-statement
    __scopes__ = scope.split(".") if scope else []


def get_scope() -> str:
    return ".".join(__scopes__)


scope = create_named_scope(__scopes__)
