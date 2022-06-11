# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

"""
Internal initialization code. Should not be called by users.
"""

import contextlib
import sys

from symforce import logger
from symforce import typing as T

from . import _sympy_count_ops


def modify_symbolic_api(sympy_module: T.Any) -> None:
    """
    Augment the sympy API for symforce use.

    Args:
        sympy_module (module):
    """
    override_symbol_new(sympy_module)
    override_simplify(sympy_module)
    override_limit(sympy_module)
    override_subs(sympy_module)
    add_scoping(sympy_module)
    add_custom_methods(sympy_module)
    override_solve(sympy_module)
    override_count_ops(sympy_module)
    override_matrix_symbol(sympy_module)
    add_derivatives(sympy_module)
    attach_symforce_modules(sympy_module)


def override_symbol_new(sympy_module: T.Any) -> None:
    """
    Override Symbol.__new__ to incorporate named scopes and default to reals instead of complex.

    Args:
        sympy_module (module):
    """
    if sympy_module.__package__ == "symengine":

        class Symbol(sympy_module.Symbol):  # type: ignore # mypy thinks sm.Symbol isn't defined
            def __init__(
                self, name: str, commutative: bool = True, real: bool = True, positive: bool = None
            ) -> None:
                scoped_name = ".".join(sympy_module.__scopes__ + [name])
                super().__init__(scoped_name, commutative=commutative, real=real, positive=positive)

                # TODO(hayk): This is not enabled, right now all symbols are commutative, real, but
                # not positive.
                # self.is_real = real
                # self.is_positive = positive
                # self.is_commutative = commutative

        sympy_module.Symbol = Symbol

        # Because we're creating a new subclass, we also need to override sm.symbols to use this one
        original_symbols = sympy_module.symbols

        def new_symbols(names: str, **args: T.Any) -> T.Union[T.Sequence[Symbol], Symbol]:
            cls = args.pop("cls", Symbol)
            return original_symbols(names, **dict(args, cls=cls))

        sympy_module.symbols = new_symbols

    else:
        assert sympy_module.__package__ == "sympy"

        # Save original
        original_symbol_new = sympy_module.Symbol.__new__

        @staticmethod  # type: ignore
        def new_symbol(
            cls: T.Any,
            name: str,
            commutative: bool = True,
            real: bool = True,
            positive: bool = None,
        ) -> None:
            name = ".".join(sympy_module.__scopes__ + [name])
            obj = original_symbol_new(
                cls, name, commutative=commutative, real=real, positive=positive
            )
            return obj

        sympy_module.Symbol.__new__ = new_symbol


def override_simplify(sympy_module: T.Type) -> None:
    """
    Override simplify so that we can use it with the symengine backend

    Args:
        sympy_module (module):
    """
    if hasattr(sympy_module, "simplify"):
        return

    import sympy

    def simplify(*args: T.Any, **kwargs: T.Any) -> sympy.Basic:
        logger.warning("Converting to sympy to use .simplify")
        return sympy_module.S(sympy.simplify(sympy.S(*args), **kwargs))

    sympy_module.simplify = simplify


def override_limit(sympy_module: T.Type) -> None:
    """
    Override limit so that we can use it with the symengine backend

    Args:
        sympy_module (module):
    """
    if hasattr(sympy_module, "limit"):
        return

    import sympy

    def limit(
        e: T.Any, z: T.Any, z0: T.Any, dir: str = "+"  # pylint: disable=redefined-builtin
    ) -> sympy.Basic:
        logger.warning("Converting to sympy to use .limit")
        return sympy_module.S(sympy.limit(sympy.S(e), sympy.S(z), sympy.S(z0), dir=dir))

    sympy_module.limit = limit


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


def add_scoping(sympy_module: T.Type) -> None:
    """
    Add name scopes to the symbolic API.
    """

    def set_scope(scope: str) -> None:
        sympy_module.__scopes__ = scope.split(".") if scope else []

    setattr(sympy_module, "set_scope", set_scope)
    setattr(sympy_module, "get_scope", lambda: ".".join(sympy_module.__scopes__))
    sympy_module.set_scope("")

    setattr(sympy_module, "scope", create_named_scope(sympy_module.__scopes__))


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
    from symforce import sympy as sm

    for key, value in subs_pairs:

        if python_util.scalar_like(key):
            assert python_util.scalar_like(value)
            new_subs_dict[key] = value
            continue

        if isinstance(key, sm.DataBuffer) or isinstance(value, sm.DataBuffer):
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


def override_subs(sympy_module: T.Type) -> None:
    """
    Patch subs to support storage classes in substitution by calling to_storage() to flatten
    the substitution dict. This has to be done slightly differently in symengine and sympy.
    """
    if sympy_module.__name__ == "symengine":
        # For some reason this doesn't exist unless we import the symengine_wrapper directly as a
        # local variable, i.e. just `import symengine.lib.symengine_wrapper` does not let us access
        # symengine.lib.symengine_wrapper
        import symengine.lib.symengine_wrapper as wrapper  # pylint: disable=no-name-in-module

        original_get_dict = wrapper.get_dict
        wrapper.get_dict = lambda *args, **kwargs: original_get_dict(
            _get_subs_dict(*args, **kwargs)
        )
    elif sympy_module.__name__ == "sympy":
        original_subs = sympy_module.Basic.subs
        sympy_module.Basic.subs = lambda self, *args, **kwargs: original_subs(
            self, _get_subs_dict(*args, **kwargs), **kwargs
        )
    else:
        raise NotImplementedError(f"Unknown backend: '{sympy_module.__name__}'")


def override_solve(sympy_module: T.Type) -> None:
    """
    Patch solve to make symengine's API consistent with SymPy's.  Currently this only supports
    solutions expressed by symengine as an sm.FiniteSet or EmptySet
    """
    if sympy_module.__name__ == "symengine":
        original_solve = sympy_module.solve

        # Unfortunately this already doesn't have a docstring or argument list in symengine
        def solve(*args: T.Any, **kwargs: T.Any) -> T.List[T.Scalar]:
            solution = original_solve(*args, **kwargs)
            from symengine.lib.symengine_wrapper import (  # pylint: disable=no-name-in-module
                EmptySet,
            )

            if isinstance(solution, sympy_module.FiniteSet):
                return list(solution.args)
            elif isinstance(solution, EmptySet):
                return []
            else:
                raise NotImplementedError(
                    f"sm.solve currently only supports FiniteSet and EmptySet results on the SymEngine backend, got {type(solution)} instead"
                )

        sympy_module.solve = solve
    elif sympy_module.__name__ == "sympy":
        # This one is fine as is
        return
    else:
        raise NotImplementedError(f"Unknown backend: '{sympy_module.__name__}'")


def override_count_ops(sympy_module: T.Type) -> None:
    """
    Patch count_ops to yield more reasonable outputs from the perspective of generated code. Only
    sympy.count_ops is modified here as the symengine.count_ops is modified directly.
    """
    if sympy_module.__name__ == "sympy":
        sympy_module.count_ops = _sympy_count_ops.count_ops


def add_custom_methods(sympy_module: T.Type) -> None:
    """
    Add safe helper methods to the symbolic API.
    """

    def register(func: T.Callable) -> T.Callable:
        setattr(sympy_module, func.__name__, func)
        return func

    # Should match C++ default epsilon in epsilon.h
    sympy_module.default_epsilon = 10 * sys.float_info.epsilon

    # Create a symbolic epsilon to encourage consistent use
    sympy_module.epsilon = sympy_module.Symbol("epsilon")

    # Save original functions to reference in wrappers
    original_atan2 = sympy_module.atan2

    @register
    def atan2(y: T.Scalar, x: T.Scalar, epsilon: T.Scalar = 0) -> T.Scalar:
        return original_atan2(y, x + (sympy_module.sign(x) + 0.5) * epsilon)

    @register
    def asin_safe(x: T.Scalar, epsilon: T.Scalar = 0) -> T.Scalar:
        x_safe = sympy_module.Max(-1 + epsilon, sympy_module.Min(1 - epsilon, x))
        return sympy_module.asin(x_safe)

    @register
    def acos_safe(x: T.Scalar, epsilon: T.Scalar = 0) -> T.Scalar:
        x_safe = sympy_module.Max(-1 + epsilon, sympy_module.Min(1 - epsilon, x))
        return sympy_module.acos(x_safe)

    @register
    def wrap_angle(x: T.Scalar) -> T.Scalar:
        """
        Wrap an angle to the interval [-pi, pi).  Commonly used to compute the shortest signed
        distance between two angles.

        See also: `angle_diff`
        """
        return sympy_module.Mod(x + sympy_module.pi, 2 * sympy_module.pi) - sympy_module.pi

    @register
    def angle_diff(x: T.Scalar, y: T.Scalar) -> T.Scalar:
        """
        Return the difference x - y, but wrapped into [-pi, pi); i.e. the angle `diff` closest to 0
        such that x = y + diff (mod 2pi).

        See also: `wrap_angle`
        """
        return sympy_module.wrap_angle(x - y)

    @register
    def sign_no_zero(x: T.Scalar) -> T.Scalar:
        """
        Returns -1 if x is negative, 1 if x is positive, and 1 if x is zero.
        """
        return 2 * sympy_module.Min(sympy_module.sign(x), 0) + 1

    @register
    def copysign_no_zero(x: T.Scalar, y: T.Scalar) -> T.Scalar:
        """
        Returns a value with the magnitude of x and sign of y. If y is zero, returns positive x.
        """
        return sympy_module.Abs(x) * sign_no_zero(y)

    @register
    def argmax_onehot(vals: T.Sequence[T.Scalar]) -> T.List[T.Scalar]:
        """
        Returns a list l such that l[i] = 1.0 if i is the smallest index such that
        vals[i] equals Max(*vals). l[i] = 0.0 otherwise.

        Precondition:
            vals has at least one element
        """
        m = sympy_module.Max(*vals)
        results = []
        have_max_already = 0
        for val in vals:
            results.append(
                sympy_module.logical_and(
                    sympy_module.is_nonnegative(val - m),
                    sympy_module.logical_not(have_max_already, unsafe=True),
                    unsafe=True,
                )
            )
            have_max_already = sympy_module.logical_or(have_max_already, results[-1], unsafe=True)
        return results

    @register
    def argmax(vals: T.Sequence[T.Scalar]) -> T.Scalar:
        """
        Returns i (as a T.Scalar) such that i is the smallest index such that
        vals[i] equals Max(*vals).

        Precondition:
            vals has at least one element
        """
        return sum(i * val for i, val in enumerate(argmax_onehot(vals)))

    from symforce import logic

    logic.add_logic_methods(sympy_module)


def override_matrix_symbol(sympy_module: T.Type) -> None:
    """
    Patch to make sympy MatrixSymbol consistent with symforce's custom Databuffer for symengine
    We want to force Databuffers to be 1-D since otherwise CSE will (rightfully) treat each index
    as a separate expression.
    """
    if sympy_module.__name__ == "sympy":
        from symforce.databuffer import DataBuffer

        DataBuffer.__sympy_module__ = sympy_module
        sympy_module.DataBuffer = DataBuffer


def add_derivatives(sympy_module: T.Type) -> None:
    """
    Add derivatives for floor, sign, and mod

    Only necessary on sympy
    """
    if sympy_module.__name__ == "sympy":
        # Hack in some key derivatives that sympy doesn't do. For all these cases the derivatives
        # here are correct except at the discrete switching point, which is correct for our
        # numerical purposes.
        setattr(sympy_module.floor, "_eval_derivative", lambda s, v: sympy_module.S.Zero)
        setattr(sympy_module.sign, "_eval_derivative", lambda s, v: sympy_module.S.Zero)

        def mod_derivative(self: T.Any, x: T.Any) -> T.Any:
            p, q = self.args
            return self._eval_rewrite_as_floor(p, q).diff(x)  # pylint: disable=protected-access

        setattr(sympy_module.Mod, "_eval_derivative", mod_derivative)


def attach_symforce_modules(sympy_module: T.Type) -> None:
    """
    Add everything in the geo and cam modules to symforce.sympy
    """

    # Purge geo and cam from modules, since they may refer to other sympy
    allowlist = {
        "symforce.typing",
        "symforce._version",
        "symforce.logic",
        "symforce.initialization",
        "symforce._sympy_count_ops",
    }

    for module_name in list(sys.modules):
        if module_name.startswith("symforce.") and module_name not in allowlist:
            del sys.modules[module_name]

    # Similarly, purge from the symforce module itself
    import symforce

    if hasattr(symforce, "geo"):
        del symforce.geo
    if hasattr(symforce, "cam"):
        del symforce.cam

    # Now, reload them
    from symforce import geo
    from symforce import cam

    # Attach everything in geo and cam to symforce.sympy
    for symforce_module in (geo, cam):
        for var_name in dir(symforce_module):
            # Except private things
            if var_name.startswith("_"):
                continue

            # And geo.Matrix
            if var_name == "Matrix":
                continue

            setattr(sympy_module, var_name, getattr(symforce_module, var_name))
